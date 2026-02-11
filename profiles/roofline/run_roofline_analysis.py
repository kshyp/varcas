#!/usr/bin/env python3
"""
Master Roofline Analysis Orchestrator

This script orchestrates both static and dynamic roofline analysis:
1. Runs static analysis (theoretical bounds)
2. Checks if vLLM is running or starts it
3. Runs dynamic profiling using ncu
4. Generates visualization report

Usage:
    # Static analysis only (no vLLM required):
    python run_roofline_analysis.py --static-only

    # Full analysis (requires vLLM to be running or will start it):
    python run_roofline_analysis.py --model TheBloke/Llama-2-7B-AWQ --profile chat_medium

    # With specific parameters:
    python run_roofline_analysis.py --model TheBloke/Llama-2-7B-AWQ \
                                     --quantization awq \
                                     --profile chat_medium \
                                     --duration 60
"""

import subprocess
import sys
import os
import json
import time
import signal
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict
import argparse


class RooflineOrchestrator:
    """Orchestrates the complete roofline analysis workflow."""
    
    def __init__(self, output_dir: Path, model: str = "TheBloke/Llama-2-7B-AWQ",
                 quantization: str = "awq", bits: int = 4):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.model = model
        self.quantization = quantization
        self.bits = bits
        self.vllm_process: Optional[subprocess.Popen] = None
        self.vllm_started_by_us = False
        
    def run_static_analysis(self) -> Optional[Path]:
        """Run static roofline analysis."""
        print("\n" + "="*70)
        print("PHASE 1: STATIC ROOFLINE ANALYSIS")
        print("="*70)
        
        script_path = Path(__file__).parent / "roofline_static.py"
        output_file = self.output_dir / "static_analysis.json"
        
        cmd = [
            sys.executable, str(script_path),
            "--model", self.model,
            "--quantization", self.quantization,
            "--bits", str(self.bits),
            "--output", str(output_file)
        ]
        
        print(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        
        if result.returncode == 0 and output_file.exists():
            print(f"✅ Static analysis complete: {output_file}")
            return output_file
        else:
            print("❌ Static analysis failed")
            return None
    
    def is_vllm_running(self, port: int = 8000) -> bool:
        """Check if vLLM is already running."""
        try:
            import urllib.request
            urllib.request.urlopen(f"http://localhost:{port}/health", timeout=2)
            return True
        except:
            return False
    
    def start_vllm(self, max_model_len: int = 2048, port: int = 8000,
                   gpu_memory_utilization: float = 0.9) -> bool:
        """Start vLLM server."""
        print("\n" + "="*70)
        print("STARTING VLLM SERVER")
        print("="*70)
        
        cmd = [
            sys.executable, "-m", "vllm.entrypoints.openai.api_server",
            "--model", self.model,
            "--dtype", "half",
            "--max-model-len", str(max_model_len),
            "--gpu-memory-utilization", str(gpu_memory_utilization),
            "--port", str(port),
            "--enforce-eager"  # Better for profiling
        ]
        
        if self.quantization:
            cmd.extend(["--quantization", self.quantization])
        
        print(f"Command: {' '.join(cmd)}")
        
        log_file = open(self.output_dir / "vllm_server.log", "w")
        
        self.vllm_process = subprocess.Popen(
            cmd,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            preexec_fn=os.setsid if sys.platform != "win32" else None
        )
        
        # Wait for server to be ready
        print("Waiting for vLLM server to be ready...")
        max_wait = 180
        start_time = time.time()
        
        while time.time() - start_time < max_wait:
            if self.is_vllm_running(port):
                elapsed = time.time() - start_time
                print(f"✅ vLLM ready after {elapsed:.1f}s")
                self.vllm_started_by_us = True
                return True
            
            if self.vllm_process.poll() is not None:
                print("❌ vLLM process terminated unexpectedly!")
                log_file.close()
                with open(self.output_dir / "vllm_server.log") as f:
                    print("Server log:")
                    print(f.read())
                return False
            
            time.sleep(1)
            if int(time.time() - start_time) % 10 == 0:
                print(f"  ... still waiting ({int(time.time() - start_time)}s)")
        
        print("❌ Timeout waiting for vLLM server")
        return False
    
    def stop_vllm(self):
        """Stop vLLM server if we started it."""
        if self.vllm_process and self.vllm_started_by_us:
            print("\nStopping vLLM server...")
            try:
                if sys.platform != "win32":
                    os.killpg(os.getpgid(self.vllm_process.pid), signal.SIGTERM)
                else:
                    self.vllm_process.terminate()
                
                self.vllm_process.wait(timeout=15)
                print("✅ vLLM stopped")
            except:
                try:
                    self.vllm_process.kill()
                    print("✅ vLLM killed")
                except:
                    print("⚠️ Could not stop vLLM")
            
            self.vllm_process = None
            self.vllm_started_by_us = False
    
    def run_load_only_test(self, profile: str, duration: int) -> Optional[Path]:
        """Run load test without NCU profiling."""
        print("\n" + "="*70)
        print("PHASE 2: LOAD TEST (without NCU)")
        print("="*70)
        
        harness_path = Path("varcas/benchmark_harness/varcas_load_harness.py")
        if not harness_path.exists():
            print(f"❌ Load harness not found: {harness_path}")
            return None
        
        output_file = self.output_dir / "load_test_results.json"
        
        cmd = [
            sys.executable, str(harness_path),
            "--profile", profile,
            "--duration", str(duration),
            "--meaningful-prompts",
            "--output", str(output_file)
        ]
        
        print(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=duration * 2)
        
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        
        if output_file.exists():
            print(f"✅ Load test complete: {output_file}")
            return output_file
        else:
            print("❌ Load test failed")
            return None
    
    def run_ncu_profiling(self, profile: str, duration: int) -> Optional[Path]:
        """Run load test with NCU profiling."""
        print("\n" + "="*70)
        print("PHASE 2: DYNAMIC PROFILING WITH NCU")
        print("="*70)
        
        harness_path = Path("varcas/benchmark_harness/varcas_load_harness.py")
        if not harness_path.exists():
            print(f"❌ Load harness not found: {harness_path}")
            return None
        
        # Check if ncu is available
        try:
            result = subprocess.run(["which", "ncu"], capture_output=True)
            if result.returncode != 0:
                print("⚠️ NCU not found, skipping dynamic profiling")
                return None
        except:
            print("⚠️ Cannot check for NCU, skipping dynamic profiling")
            return None
        
        ncu_rep_file = self.output_dir / "profile.ncu-rep"
        csv_file = self.output_dir / "ncu_profile.csv"
        
        # Build NCU command
        # Note: We'll profile the entire load test run
        cmd = [
            "ncu",
            "--export", str(ncu_rep_file.with_suffix('')),
            "--force-overwrite",
            "--target-processes", "all",
            "--replay-mode", "kernel",
            "python", str(harness_path),
            "--profile", profile,
            "--duration", str(duration),
            "--meaningful-prompts",
            "--output", str(self.output_dir / "load_test_results.json")
        ]
        
        print(f"Running NCU profiling (this may take {duration * 2}s)...")
        print(f"Command: ncu --export {ncu_rep_file} ...")
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=duration * 3
            )
            
            if ncu_rep_file.exists():
                print(f"✅ NCU profiling complete: {ncu_rep_file}")
                
                # Export to CSV for easier parsing
                try:
                    subprocess.run([
                        "ncu", "--import", str(ncu_rep_file),
                        "--csv", "--page", "details"
                    ], stdout=open(csv_file, 'w'), stderr=subprocess.DEVNULL, check=True)
                    print(f"✅ Exported to CSV: {csv_file}")
                    return csv_file
                except:
                    return ncu_rep_file
            else:
                print("❌ NCU output not found")
                print("STDOUT:", result.stdout[-2000:] if len(result.stdout) > 2000 else result.stdout)
                print("STDERR:", result.stderr[-2000:] if len(result.stderr) > 2000 else result.stderr)
                return None
                
        except subprocess.TimeoutExpired:
            print("❌ NCU profiling timed out")
            return None
        except Exception as e:
            print(f"❌ NCU profiling failed: {e}")
            return None
    
    def parse_and_analyze_ncu(self, ncu_file: Path) -> Optional[Path]:
        """Parse NCU output and generate dynamic analysis."""
        print("\n" + "="*70)
        print("PHASE 3: ANALYZING NCU DATA")
        print("="*70)
        
        # This is a simplified analysis - just load the data we have
        # In a full implementation, we'd parse the NCU output in detail
        
        analysis = {
            "analysis_type": "dynamic_roofline",
            "timestamp": datetime.now().isoformat(),
            "ncu_file": str(ncu_file),
            "note": "Raw NCU data available for manual analysis"
        }
        
        # Try to extract basic info from CSV if available
        if ncu_file.suffix == '.csv':
            try:
                import csv
                kernels = []
                with open(ncu_file) as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        if 'Kernel Name' in row:
                            kernels.append({
                                'name': row['Kernel Name'],
                                'duration': row.get('GPU Time [us]', '0'),
                            })
                
                analysis['kernels_found'] = len(kernels)
                analysis['top_kernels'] = kernels[:20] if kernels else []
                print(f"✅ Parsed {len(kernels)} kernels from NCU CSV")
            except Exception as e:
                print(f"⚠️ Could not parse NCU CSV: {e}")
        
        output_file = self.output_dir / "dynamic_analysis.json"
        with open(output_file, 'w') as f:
            json.dump(analysis, f, indent=2)
        
        print(f"✅ Dynamic analysis saved: {output_file}")
        return output_file
    
    def generate_report(self, static_file: Optional[Path], 
                       dynamic_file: Optional[Path]) -> Optional[Path]:
        """Generate visualization report."""
        print("\n" + "="*70)
        print("PHASE 4: GENERATING VISUALIZATION REPORT")
        print("="*70)
        
        script_path = Path(__file__).parent / "visualize_roofline.py"
        
        cmd = [sys.executable, str(script_path)]
        
        if static_file:
            cmd.extend(["--static", str(static_file)])
        if dynamic_file:
            cmd.extend(["--dynamic", str(dynamic_file)])
        
        report_file = self.output_dir / "roofline_report.html"
        cmd.extend(["--output", str(report_file)])
        
        print(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        
        if report_file.exists():
            print(f"✅ Report generated: {report_file}")
            return report_file
        else:
            print("❌ Report generation failed")
            return None
    
    def run_full_analysis(self, profile: str = "chat_medium", duration: int = 30,
                         skip_dynamic: bool = False, manage_vllm: bool = True) -> Dict:
        """Run complete roofline analysis."""
        
        results = {
            "timestamp": datetime.now().isoformat(),
            "output_dir": str(self.output_dir),
            "phases": {}
        }
        
        try:
            # Phase 1: Static analysis (always run)
            static_file = self.run_static_analysis()
            results["phases"]["static"] = {"file": str(static_file) if static_file else None}
            
            if skip_dynamic:
                print("\n⚠️ Skipping dynamic analysis as requested")
                results["phases"]["dynamic"] = {"skipped": True}
            else:
                # Check/start vLLM
                vllm_was_running = self.is_vllm_running()
                
                if not vllm_was_running:
                    if manage_vllm:
                        if not self.start_vllm():
                            results["phases"]["dynamic"] = {"error": "Failed to start vLLM"}
                            # Still generate report with static only
                            report_file = self.generate_report(static_file, None)
                            results["report"] = str(report_file) if report_file else None
                            return results
                    else:
                        print("⚠️ vLLM not running and --no-manage-vllm specified")
                        results["phases"]["dynamic"] = {"skipped": "vLLM not running"}
                        report_file = self.generate_report(static_file, None)
                        results["report"] = str(report_file) if report_file else None
                        return results
                else:
                    print("✅ Using existing vLLM server")
                
                try:
                    # Phase 2: Dynamic profiling
                    # Try NCU first, fall back to load-only
                    dynamic_file = None
                    
                    # Check if we can use NCU (requires specific permissions/setup)
                    ncu_check = subprocess.run(
                        ["ncu", "--version"],
                        capture_output=True
                    )
                    
                    if ncu_check.returncode == 0:
                        print("NCU available, attempting profiling...")
                        ncu_file = self.run_ncu_profiling(profile, duration)
                        if ncu_file:
                            dynamic_file = self.parse_and_analyze_ncu(ncu_file)
                    
                    if not dynamic_file:
                        # Fallback: just run load test to get metrics
                        print("Running load test without NCU...")
                        load_file = self.run_load_only_test(profile, duration)
                        if load_file:
                            # Create minimal dynamic analysis from load test
                            dynamic_file = self.output_dir / "dynamic_analysis.json"
                            with open(load_file) as f:
                                load_data = json.load(f)
                            
                            analysis = {
                                "analysis_type": "dynamic_from_load_test",
                                "timestamp": datetime.now().isoformat(),
                                "load_test_results": load_data,
                                "note": "Load test metrics without kernel-level profiling"
                            }
                            with open(dynamic_file, 'w') as f:
                                json.dump(analysis, f, indent=2)
                    
                    results["phases"]["dynamic"] = {"file": str(dynamic_file) if dynamic_file else None}
                    
                finally:
                    # Only stop vLLM if we started it
                    if self.vllm_started_by_us:
                        self.stop_vllm()
            
            # Phase 4: Generate report
            static_path = static_file if static_file else None
            dynamic_path = Path(results["phases"]["dynamic"].get("file")) if "dynamic" in results["phases"] and results["phases"]["dynamic"].get("file") else None
            
            report_file = self.generate_report(static_path, dynamic_path)
            results["report"] = str(report_file) if report_file else None
            
        except Exception as e:
            print(f"\n❌ Analysis failed: {e}")
            import traceback
            traceback.print_exc()
            results["error"] = str(e)
        
        return results


def main():
    parser = argparse.ArgumentParser(
        description="Master Roofline Analysis for vLLM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Static analysis only (fast, no GPU required):
    python run_roofline_analysis.py --static-only

    # Full analysis with auto-managed vLLM:
    python run_roofline_analysis.py --model TheBloke/Llama-2-7B-AWQ --profile chat_medium

    # Analysis with existing vLLM server:
    python run_roofline_analysis.py --profile chat_medium --no-manage-vllm

    # Longer profiling duration:
    python run_roofline_analysis.py --profile chat_high --duration 120
        """
    )
    
    parser.add_argument("--model", default="TheBloke/Llama-2-7B-AWQ",
                       help="Model name for analysis")
    parser.add_argument("--quantization", default="awq",
                       help="Quantization type (awq, gptq, fp16, etc.)")
    parser.add_argument("--bits", type=int, default=4,
                       help="Bits per parameter")
    parser.add_argument("--profile", default="chat_medium",
                       choices=["chat_low", "chat_medium", "chat_high",
                               "rag_small", "rag_medium", "rag_large",
                               "code_low", "code_medium", "code_high"],
                       help="Load profile from varcas harness")
    parser.add_argument("--duration", type=int, default=30,
                       help="Profiling duration in seconds")
    parser.add_argument("--output-dir", default="varcas/profiles/roofline",
                       help="Output directory for results")
    parser.add_argument("--static-only", action="store_true",
                       help="Run only static analysis (no vLLM required)")
    parser.add_argument("--no-manage-vllm", action="store_true",
                       help="Don't start/stop vLLM (use existing server)")
    parser.add_argument("--max-model-len", type=int, default=2048,
                       help="Maximum model context length")
    
    args = parser.parse_args()
    
    # Create timestamped output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / f"analysis_{timestamp}"
    
    print("="*70)
    print("ROOFLINE ANALYSIS FOR VLLM")
    print("="*70)
    print(f"Model: {args.model}")
    print(f"Profile: {args.profile}")
    print(f"Output: {output_dir}")
    print(f"Static only: {args.static_only}")
    print("="*70)
    
    # Run analysis
    orchestrator = RooflineOrchestrator(
        output_dir=output_dir,
        model=args.model,
        quantization=args.quantization,
        bits=args.bits
    )
    
    results = orchestrator.run_full_analysis(
        profile=args.profile,
        duration=args.duration,
        skip_dynamic=args.static_only,
        manage_vllm=not args.no_manage_vllm
    )
    
    # Save summary
    summary_file = output_dir / "analysis_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Final summary
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    print(f"\nOutput directory: {output_dir}")
    print(f"\nKey files:")
    print(f"  - Summary:     {summary_file}")
    
    if results["phases"].get("static", {}).get("file"):
        print(f"  - Static:      {results['phases']['static']['file']}")
    if results["phases"].get("dynamic", {}).get("file"):
        print(f"  - Dynamic:     {results['phases']['dynamic']['file']}")
    if results.get("report"):
        print(f"  - Report:      {results['report']}")
        print(f"\n👉 Open {results['report']} in your browser to view the results")
    
    print("\n" + "="*70)


if __name__ == "__main__":
    main()
