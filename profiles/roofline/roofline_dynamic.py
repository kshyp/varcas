#!/usr/bin/env python3
"""
Dynamic Roofline Analysis for vLLM using NCU Profiling

This script:
1. Starts vLLM server with the specified model
2. Runs the varcas load harness to generate traffic
3. Uses ncu to profile GPU kernels during execution
4. Analyzes the profiling data to create a roofline plot
5. Compares actual performance with theoretical bounds

Usage:
    python roofline_dynamic.py --model TheBloke/Llama-2-7B-AWQ \
                               --profile chat_medium \
                               --duration 60
"""

import subprocess
import json
import time
import signal
import sys
import os
import argparse
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
import tempfile
import shutil


@dataclass
class NCUKernelsMetrics:
    """Metrics extracted from NCU profiling."""
    kernel_name: str
    grid_size: Tuple[int, int, int]
    block_size: Tuple[int, int, int]
    
    # Performance counters
    duration_ms: float
    sm_active_pct: float
    
    # Memory metrics
    dram_read_bytes: int
    dram_write_bytes: int
    l2_read_bytes: int
    l2_write_bytes: int
    shared_mem_bytes: int
    
    # Compute metrics
    flop_count_sp: int  # FP32 FLOPs
    flop_count_hp: int  # FP16 FLOPs
    
    @property
    def total_flops(self) -> int:
        return self.flop_count_sp + self.flop_count_hp
    
    @property
    def total_bytes(self) -> int:
        return self.dram_read_bytes + self.dram_write_bytes
    
    @property
    def arithmetic_intensity(self) -> float:
        return self.total_flops / self.total_bytes if self.total_bytes > 0 else 0
    
    @property
    def achieved_tflops(self) -> float:
        return self.total_flops / (self.duration_ms * 1e-3) / 1e12


class DynamicRooflineAnalyzer:
    """
    Dynamic roofline analyzer that profiles actual execution.
    """
    
    def __init__(self, output_dir: Path, gpu_name: str = "Tesla T4"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.gpu_name = gpu_name
        self.vllm_process: Optional[subprocess.Popen] = None
        
        # GPU specs (from static analysis)
        self.gpu_specs = self._get_gpu_specs(gpu_name)
        
    def _get_gpu_specs(self, name: str) -> Dict:
        """Get GPU specifications."""
        specs = {
            "Tesla T4": {
                "peak_fp16_flops": 65.0,  # TFLOPS
                "peak_fp32_flops": 8.1,
                "memory_bw": 320.0,  # GB/s
                "memory_size": 16.0,
            },
            "A100": {
                "peak_fp16_flops": 312.0,
                "peak_fp32_flops": 19.5,
                "memory_bw": 1555.0,
                "memory_size": 40.0,
            },
            "L4": {
                "peak_fp16_flops": 121.0,
                "peak_fp32_flops": 30.3,
                "memory_bw": 300.0,
                "memory_size": 24.0,
            }
        }
        
        for key in specs:
            if key in name:
                return specs[key]
        
        return specs["Tesla T4"]  # Default
    
    def start_vllm(self, model: str, quantization: Optional[str] = None,
                   max_model_len: int = 2048, port: int = 8000) -> bool:
        """Start vLLM server."""
        cmd = [
            "python", "-m", "vllm.entrypoints.openai.api_server",
            "--model", model,
            "--dtype", "half",
            "--max-model-len", str(max_model_len),
            "--gpu-memory-utilization", "0.90",
            "--port", str(port),
            "--enforce-eager"  # Disable CUDA graph for better profiling
        ]
        
        if quantization:
            cmd.extend(["--quantization", quantization])
        
        print(f"Starting vLLM: {' '.join(cmd)}")
        
        log_file = open(self.output_dir / "vllm_server.log", "w")
        
        self.vllm_process = subprocess.Popen(
            cmd,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            preexec_fn=os.setsid if sys.platform != "win32" else None
        )
        
        # Wait for server to be ready
        print("Waiting for vLLM server to be ready...")
        max_wait = 120
        start_time = time.time()
        
        while time.time() - start_time < max_wait:
            try:
                import urllib.request
                urllib.request.urlopen(f"http://localhost:{port}/health", timeout=1)
                print(f"vLLM server ready after {time.time() - start_time:.1f}s")
                return True
            except:
                time.sleep(1)
                if self.vllm_process.poll() is not None:
                    print("vLLM process terminated unexpectedly!")
                    return False
        
        print("Timeout waiting for vLLM server")
        return False
    
    def stop_vllm(self):
        """Stop vLLM server."""
        if self.vllm_process:
            print("Stopping vLLM server...")
            try:
                if sys.platform != "win32":
                    os.killpg(os.getpgid(self.vllm_process.pid), signal.SIGTERM)
                else:
                    self.vllm_process.terminate()
                
                self.vllm_process.wait(timeout=10)
            except:
                try:
                    self.vllm_process.kill()
                except:
                    pass
            
            self.vllm_process = None
    
    def run_load_test(self, profile: str = "chat_medium", duration: int = 30,
                      meaningful: bool = True) -> bool:
        """Run varcas load harness."""
        harness_path = Path("varcas/benchmark_harness/varcas_load_harness.py")
        
        if not harness_path.exists():
            print(f"Load harness not found at {harness_path}")
            return False
        
        cmd = [
            "python", str(harness_path),
            "--profile", profile,
            "--duration", str(duration),
            "--output", str(self.output_dir / "load_test_results.json")
        ]
        
        if meaningful:
            cmd.append("--meaningful-prompts")
        
        print(f"Running load test: {' '.join(cmd)}")
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"Load test failed: {result.stderr}")
            return False
        
        print("Load test completed successfully")
        return True
    
    def profile_with_ncu(self, profile: str = "chat_medium", duration: int = 30,
                         metrics: Optional[List[str]] = None) -> Optional[Path]:
        """
        Profile vLLM execution with ncu.
        
        Uses ncu to collect detailed kernel metrics during load test.
        """
        if metrics is None:
            # Key metrics for roofline analysis
            metrics = [
                "sm__throughput.avg.pct_of_peak_sustained_elapsed",
                "dram__bytes_read.sum",
                "dram__bytes_write.sum",
                "lts__t_bytes_equiv_l1sectormiss_pipe_lsu_mem_global_op_ld.sum",
                "lts__t_bytes_equiv_l1sectormiss_pipe_lsu_mem_global_op_st.sum",
                "smsp__sass_thread_inst_executed_op_fadd_pred_on.sum",
                "smsp__sass_thread_inst_executed_op_fmul_pred_on.sum",
                "smsp__sass_thread_inst_executed_op_ffma_pred_on.sum",
                "smsp__sass_thread_inst_executed_op_hadd_pred_on.sum",
                "smsp__sass_thread_inst_executed_op_hmul_pred_on.sum",
                "smsp__sass_thread_inst_executed_op_hfma_pred_on.sum",
                "gpu__time_duration.avg",
                "launch__grid_size",
                "launch__block_size"
            ]
        
        output_file = self.output_dir / "ncu_profile.ncu-rep"
        csv_file = self.output_dir / "ncu_profile.csv"
        
        # Build ncu command with Python load harness
        harness_path = Path("varcas/benchmark_harness/varcas_load_harness.py")
        
        ncu_cmd = [
            "ncu",
            "--export", str(output_file.with_suffix('')),
            "--force-overwrite",
            "--target-processes", "all",
            "--replay-mode", "kernel",
            "--metrics", ",".join(metrics),
            "python", str(harness_path),
            "--profile", profile,
            "--duration", str(duration),
            "--meaningful-prompts",
            "--output", str(self.output_dir / "load_test_results.json")
        ]
        
        print(f"Running NCU profiling...")
        print(f"Command: {' '.join(ncu_cmd[:10])}...")
        
        try:
            result = subprocess.run(
                ncu_cmd,
                capture_output=True,
                text=True,
                timeout=duration * 3  # Generous timeout
            )
            
            # Also export to CSV for easier parsing
            if output_file.exists():
                self._export_ncu_to_csv(output_file, csv_file)
                return csv_file
            
            # Try to find the output file
            possible_outputs = list(self.output_dir.glob("*.ncu-rep"))
            if possible_outputs:
                self._export_ncu_to_csv(possible_outputs[0], csv_file)
                return csv_file
            
            print("NCU profiling output not found")
            return None
            
        except subprocess.TimeoutExpired:
            print("NCU profiling timed out")
            return None
        except Exception as e:
            print(f"NCU profiling failed: {e}")
            return None
    
    def _export_ncu_to_csv(self, ncu_rep: Path, csv_output: Path):
        """Export ncu-rep file to CSV."""
        try:
            subprocess.run([
                "ncu", "--import", str(ncu_rep),
                "--csv", "--page", "details"
            ], stdout=open(csv_output, 'w'), stderr=subprocess.DEVNULL, check=True)
        except:
            pass
    
    def parse_ncu_csv(self, csv_file: Path) -> List[NCUKernelsMetrics]:
        """Parse NCU CSV output into kernel metrics."""
        kernels = []
        
        try:
            import csv
            with open(csv_file, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    try:
                        kernel = NCUKernelsMetrics(
                            kernel_name=row.get('Kernel Name', 'unknown'),
                            grid_size=(1, 1, 1),  # Parse from row if available
                            block_size=(1, 1, 1),
                            duration_ms=float(row.get('GPU Time [us]', 0)) / 1000,
                            sm_active_pct=float(row.get('SM [%]', 0)),
                            dram_read_bytes=int(float(row.get('DRAM Bytes Read', 0))),
                            dram_write_bytes=int(float(row.get('DRAM Bytes Write', 0))),
                            l2_read_bytes=0,
                            l2_write_bytes=0,
                            shared_mem_bytes=0,
                            flop_count_sp=int(float(row.get('FLOPs (SP)', 0))),
                            flop_count_hp=int(float(row.get('FLOPs (HP)', 0)))
                        )
                        kernels.append(kernel)
                    except (ValueError, KeyError) as e:
                        continue
        except Exception as e:
            print(f"Error parsing NCU CSV: {e}")
        
        return kernels
    
    def parse_ncu_output(self, text_output: str) -> List[Dict]:
        """Parse ncu text output for kernel metrics."""
        kernels = []
        
        # Pattern matching for common NCU output formats
        kernel_pattern = re.compile(
            r'\s*([\w_<>]+)\s+'  # Kernel name
            r'([\d\.]+)\s+'     # Duration
            r'([\d\.]+)\s+'     # Metric 1
            r'([\d\.]+)',       # Metric 2
            re.MULTILINE
        )
        
        for match in kernel_pattern.finditer(text_output):
            kernels.append({
                'name': match.group(1),
                'duration': float(match.group(2)),
            })
        
        return kernels
    
    def analyze_kernels(self, kernels: List[NCUKernelsMetrics]) -> Dict:
        """
        Analyze kernel metrics for roofline characterization.
        """
        if not kernels:
            return {}
        
        # Aggregate metrics
        total_flops = sum(k.total_flops for k in kernels)
        total_bytes = sum(k.total_bytes for k in kernels)
        total_duration = sum(k.duration_ms for k in kernels)
        
        # Categorize kernels
        gemm_kernels = [k for k in kernels if any(x in k.kernel_name.lower() 
                        for x in ['gemm', 'gemv', 'cutlass', 'cublas'])]
        attn_kernels = [k for k in kernels if any(x in k.kernel_name.lower()
                        for x in ['flash', 'attention', 'softmax', 'sdpa'])]
        mem_kernels = [k for k in kernels if any(x in k.kernel_name.lower()
                       for x in ['memcpy', 'copy', 'permute', 'transpose'])]
        
        analysis = {
            "summary": {
                "total_kernels": len(kernels),
                "total_flops": total_flops,
                "total_bytes": total_bytes,
                "total_duration_ms": total_duration,
                "overall_ai": total_flops / total_bytes if total_bytes > 0 else 0,
                "overall_tflops": total_flops / (total_duration * 1e-3) / 1e12
            },
            "kernel_breakdown": {
                "gemm": self._analyze_kernel_group(gemm_kernels),
                "attention": self._analyze_kernel_group(attn_kernels),
                "memory": self._analyze_kernel_group(mem_kernels),
                "other": self._analyze_kernel_group(
                    [k for k in kernels if k not in gemm_kernels + attn_kernels + mem_kernels]
                )
            },
            "top_kernels": []
        }
        
        # Top kernels by duration
        sorted_kernels = sorted(kernels, key=lambda k: k.duration_ms, reverse=True)[:10]
        for k in sorted_kernels:
            analysis["top_kernels"].append({
                "name": k.kernel_name,
                "duration_ms": k.duration_ms,
                "flops": k.total_flops,
                "bytes": k.total_bytes,
                "ai": k.arithmetic_intensity,
                "tflops": k.achieved_tflops
            })
        
        return analysis
    
    def _analyze_kernel_group(self, kernels: List[NCUKernelsMetrics]) -> Dict:
        """Analyze a group of kernels."""
        if not kernels:
            return {"count": 0}
        
        total_flops = sum(k.total_flops for k in kernels)
        total_bytes = sum(k.total_bytes for k in kernels)
        total_duration = sum(k.duration_ms for k in kernels)
        
        return {
            "count": len(kernels),
            "total_duration_ms": total_duration,
            "total_flops": total_flops,
            "total_bytes": total_bytes,
            "avg_ai": total_flops / total_bytes if total_bytes > 0 else 0,
            "avg_tflops": total_flops / (total_duration * 1e-3) / 1e12 if total_duration > 0 else 0
        }
    
    def generate_roofline_data(self, kernel_analysis: Dict) -> Dict:
        """Generate roofline plot data from kernel analysis."""
        
        # Theoretical roofline
        peak_flops = self.gpu_specs["peak_fp16_flops"]
        mem_bw = self.gpu_specs["memory_bw"]
        ridge_point = peak_flops / mem_bw
        
        roofline = {
            "gpu_name": self.gpu_name,
            "peak_tflops": peak_flops,
            "memory_bw_gbps": mem_bw,
            "ridge_point": ridge_point,
            "theoretical_curve": [],
            "measured_points": []
        }
        
        # Generate theoretical curve
        import math
        for i in range(100):
            ai = 10 ** (-1 + i * 3 / 100)  # 0.1 to 1000
            mem_bound = ai * mem_bw / 1000  # Convert to TFLOPS
            roofline["theoretical_curve"].append({
                "ai": ai,
                "tflops": min(mem_bound, peak_flops)
            })
        
        # Add measured kernel points
        if "top_kernels" in kernel_analysis:
            for k in kernel_analysis["top_kernels"]:
                roofline["measured_points"].append({
                    "name": k["name"][:50],  # Truncate long names
                    "ai": k["ai"],
                    "tflops": k["tflops"],
                    "duration_ms": k["duration_ms"]
                })
        
        # Add aggregate points
        if "kernel_breakdown" in kernel_analysis:
            for category, data in kernel_analysis["kernel_breakdown"].items():
                if data.get("count", 0) > 0 and "avg_ai" in data:
                    roofline["measured_points"].append({
                        "name": f"[AGGREGATE] {category}",
                        "ai": data["avg_ai"],
                        "tflops": data["avg_tflops"],
                        "duration_ms": data["total_duration_ms"]
                    })
        
        return roofline
    
    def run_full_analysis(self, model: str, quantization: str, 
                          profile: str, duration: int,
                          max_model_len: int = 2048) -> Dict:
        """
        Run complete dynamic roofline analysis.
        
        Steps:
        1. Start vLLM server
        2. Run NCU profiling with load test
        3. Analyze results
        4. Generate roofline data
        """
        results = {
            "analysis_type": "dynamic_roofline",
            "timestamp": datetime.now().isoformat(),
            "config": {
                "model": model,
                "quantization": quantization,
                "profile": profile,
                "duration": duration,
                "gpu": self.gpu_name
            }
        }
        
        # Step 1: Start vLLM
        print("="*60)
        print("STEP 1: Starting vLLM server")
        print("="*60)
        
        if not self.start_vllm(model, quantization, max_model_len):
            results["error"] = "Failed to start vLLM"
            return results
        
        try:
            # Wait a bit for server to fully initialize
            time.sleep(5)
            
            # Step 2: Run NCU profiling
            print("\n" + "="*60)
            print("STEP 2: Running NCU profiling")
            print("="*60)
            
            csv_file = self.profile_with_ncu(profile, duration)
            
            if csv_file and csv_file.exists():
                # Step 3: Parse and analyze
                print("\n" + "="*60)
                print("STEP 3: Analyzing kernel metrics")
                print("="*60)
                
                kernels = self.parse_ncu_csv(csv_file)
                
                if kernels:
                    analysis = self.analyze_kernels(kernels)
                    roofline_data = self.generate_roofline_data(analysis)
                    
                    results["kernel_analysis"] = analysis
                    results["roofline"] = roofline_data
                    
                    # Print summary
                    print("\nKernel Analysis Summary:")
                    print(f"  Total kernels: {analysis['summary']['total_kernels']}")
                    print(f"  Total FLOPs: {analysis['summary']['total_flops']:.2e}")
                    print(f"  Total bytes: {analysis['summary']['total_bytes']:.2e}")
                    print(f"  Overall AI: {analysis['summary']['overall_ai']:.2f}")
                    print(f"  Achieved TFLOPS: {analysis['summary']['overall_tflops']:.2f}")
                    print(f"  Peak TFLOPS: {self.gpu_specs['peak_fp16_flops']:.2f}")
                    print(f"  Utilization: {analysis['summary']['overall_tflops'] / self.gpu_specs['peak_fp16_flops'] * 100:.1f}%")
                    
                    print("\nTop kernels by duration:")
                    for k in analysis['top_kernels'][:5]:
                        print(f"  {k['name'][:50]:50} {k['duration_ms']:.2f}ms AI={k['ai']:.2f}")
                else:
                    results["warning"] = "No kernels parsed from NCU output"
                    print("Warning: No kernels parsed")
            else:
                results["warning"] = "NCU profiling did not produce output"
                print("Warning: NCU profiling failed or produced no output")
            
            # Get load test results
            results_file = self.output_dir / "load_test_results.json"
            if results_file.exists():
                with open(results_file) as f:
                    results["load_test_results"] = json.load(f)
        
        finally:
            # Always stop vLLM
            self.stop_vllm()
        
        # Save results
        output_file = self.output_dir / "dynamic_analysis.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved to: {output_file}")
        
        return results


def main():
    parser = argparse.ArgumentParser(description="Dynamic Roofline Analysis for vLLM")
    parser.add_argument("--model", default="TheBloke/Llama-2-7B-AWQ",
                       help="Model to profile")
    parser.add_argument("--quantization", default="awq",
                       help="Quantization type")
    parser.add_argument("--profile", default="chat_medium",
                       choices=["chat_low", "chat_medium", "chat_high",
                               "rag_small", "rag_medium", "rag_large",
                               "code_low", "code_medium", "code_high"],
                       help="Load profile")
    parser.add_argument("--duration", type=int, default=30,
                       help="Profiling duration in seconds")
    parser.add_argument("--max-model-len", type=int, default=2048,
                       help="Maximum model context length")
    parser.add_argument("--output-dir", default="varcas/profiles/roofline",
                       help="Output directory")
    parser.add_argument("--skip-server", action="store_true",
                       help="Skip starting vLLM (assume it's already running)")
    
    args = parser.parse_args()
    
    # Detect GPU
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True
        )
        gpu_name = result.stdout.strip()
    except:
        gpu_name = "Tesla T4"
    
    print("="*60)
    print("Dynamic Roofline Analysis for vLLM")
    print("="*60)
    print(f"Model: {args.model}")
    print(f"Profile: {args.profile}")
    print(f"GPU: {gpu_name}")
    print(f"Duration: {args.duration}s")
    print("="*60)
    
    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / f"dynamic_{timestamp}"
    
    # Run analysis
    analyzer = DynamicRooflineAnalyzer(output_dir, gpu_name)
    
    if args.skip_server:
        # Just run load test and profile
        csv_file = analyzer.profile_with_ncu(args.profile, args.duration)
        if csv_file:
            kernels = analyzer.parse_ncu_csv(csv_file)
            analysis = analyzer.analyze_kernels(kernels)
            roofline_data = analyzer.generate_roofline_data(analysis)
            
            results = {
                "analysis_type": "dynamic_roofline",
                "timestamp": datetime.now().isoformat(),
                "kernel_analysis": analysis,
                "roofline": roofline_data
            }
            
            with open(output_dir / "dynamic_analysis.json", 'w') as f:
                json.dump(results, f, indent=2)
    else:
        results = analyzer.run_full_analysis(
            args.model, args.quantization,
            args.profile, args.duration,
            args.max_model_len
        )
    
    print("\n" + "="*60)
    print("Analysis complete!")
    print(f"Results: {output_dir}")
    print("="*60)


if __name__ == "__main__":
    main()
