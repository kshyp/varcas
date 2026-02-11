#!/usr/bin/env python3
"""
Batch Size Optimization Comparison

This script compares vLLM performance with different batch size configurations:
- Baseline: Default batching (typically 1-4)
- Optimized: Increased max_num_seqs (8, 16, 32)

Usage:
    python batch_size_optimization.py --model TheBloke/Llama-2-7B-AWQ --profile chat_medium
"""

import subprocess
import sys
import os
import json
import time
import signal
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import argparse


class BatchSizeOptimizer:
    """Compare performance across different batch size configurations."""
    
    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.vllm_process: Optional[subprocess.Popen] = None
        
    def start_vllm(self, model: str, quantization: str, max_num_seqs: int,
                   max_model_len: int = 2048, port: int = 8000) -> bool:
        """Start vLLM server with specific batch size configuration."""
        cmd = [
            sys.executable, "-m", "vllm.entrypoints.openai.api_server",
            "--model", model,
            "--dtype", "half",
            "--max-model-len", str(max_model_len),
            "--max-num-seqs", str(max_num_seqs),
            "--gpu-memory-utilization", "0.90",
            "--port", str(port),
            "--enforce-eager"
        ]
        
        if quantization:
            cmd.extend(["--quantization", quantization])
        
        print(f"\nStarting vLLM with max_num_seqs={max_num_seqs}")
        print(f"Command: {' '.join(cmd)}")
        
        log_file = open(self.output_dir / f"vllm_server_b{max_num_seqs}.log", "w")
        
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
            try:
                import urllib.request
                urllib.request.urlopen(f"http://localhost:{port}/health", timeout=2)
                elapsed = time.time() - start_time
                print(f"✅ vLLM ready after {elapsed:.1f}s")
                return True
            except:
                pass
            
            if self.vllm_process.poll() is not None:
                print("❌ vLLM process terminated unexpectedly!")
                return False
            
            time.sleep(1)
            if int(time.time() - start_time) % 20 == 0:
                print(f"  ... still waiting ({int(time.time() - start_time)}s)")
        
        print("❌ Timeout waiting for vLLM server")
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
                
                self.vllm_process.wait(timeout=15)
                print("✅ vLLM stopped")
            except:
                try:
                    self.vllm_process.kill()
                except:
                    pass
            
            self.vllm_process = None
    
    def run_load_test(self, profile: str, duration: int, batch_size: int) -> Optional[Dict]:
        """Run load test and return metrics."""
        harness_path = Path("varcas/benchmark_harness/varcas_load_harness.py")
        
        if not harness_path.exists():
            print(f"❌ Load harness not found: {harness_path}")
            return None
        
        output_file = self.output_dir / f"results_b{batch_size}.json"
        
        cmd = [
            sys.executable, str(harness_path),
            "--profile", profile,
            "--duration", str(duration),
            "--meaningful-prompts",
            "--output", str(output_file)
        ]
        
        print(f"\nRunning load test (batch={batch_size}): {' '.join(cmd)}")
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=duration * 2 + 30
            )
            
            print(result.stdout)
            
            if output_file.exists():
                with open(output_file) as f:
                    data = json.load(f)
                
                metrics = data.get("metrics", {})
                return {
                    "batch_size": batch_size,
                    "total_requests": metrics.get("total_requests", 0),
                    "successful_requests": metrics.get("successful_requests", 0),
                    "throughput_rps": metrics.get("throughput_rps", 0),
                    "throughput_tok_s": metrics.get("throughput_tok_s", 0),
                    "ttft_p50_ms": metrics.get("ttft_p50_ms", 0),
                    "ttft_p99_ms": metrics.get("ttft_p99_ms", 0),
                    "tpot_p50_ms": metrics.get("tpot_p50_ms", 0),
                    "tpot_p99_ms": metrics.get("tpot_p99_ms", 0),
                    "latency_p50_ms": metrics.get("latency_p50_ms", 0),
                    "latency_p99_ms": metrics.get("latency_p99_ms", 0),
                    "raw_data": data
                }
            else:
                print("❌ Load test output not found")
                return None
                
        except subprocess.TimeoutExpired:
            print("❌ Load test timed out")
            return None
        except Exception as e:
            print(f"❌ Load test failed: {e}")
            return None
    
    def run_comparison(self, model: str, quantization: str, profile: str,
                      duration: int, batch_sizes: List[int]) -> Dict:
        """Run comparison across multiple batch sizes."""
        results = {
            "timestamp": datetime.now().isoformat(),
            "model": model,
            "quantization": quantization,
            "profile": profile,
            "configurations": []
        }
        
        for batch_size in batch_sizes:
            print("\n" + "="*70)
            print(f"TESTING BATCH SIZE: {batch_size}")
            print("="*70)
            
            # Start vLLM with this batch size
            if not self.start_vllm(model, quantization, batch_size):
                print(f"❌ Failed to start vLLM with batch_size={batch_size}")
                continue
            
            try:
                # Wait for server to fully initialize
                time.sleep(3)
                
                # Run load test
                metrics = self.run_load_test(profile, duration, batch_size)
                
                if metrics:
                    results["configurations"].append(metrics)
                    print(f"\n✅ Completed batch_size={batch_size}")
                else:
                    print(f"\n❌ Failed to get metrics for batch_size={batch_size}")
                    
            finally:
                self.stop_vllm()
                time.sleep(2)  # Give time for cleanup
        
        return results
    
    def analyze_results(self, results: Dict) -> Dict:
        """Analyze and compare results across batch sizes."""
        if not results.get("configurations"):
            return {"error": "No valid configurations to analyze"}
        
        analysis = {
            "summary": {
                "best_throughput": None,
                "best_latency": None,
                "recommendation": None
            },
            "comparison": []
        }
        
        # Sort by batch size
        configs = sorted(results["configurations"], key=lambda x: x["batch_size"])
        
        print("\n" + "="*70)
        print("PERFORMANCE COMPARISON")
        print("="*70)
        print(f"\n{'Batch':<8} {'Req/s':<10} {'Tok/s':<12} {'TTFT p50':<12} {'TPOT p50':<12} {'Latency p50':<12}")
        print("-"*70)
        
        max_throughput = 0
        min_latency = float('inf')
        best_throughput_config = None
        best_latency_config = None
        
        for config in configs:
            batch = config["batch_size"]
            rps = config["throughput_rps"]
            tok_s = config["throughput_tok_s"]
            ttft = config["ttft_p50_ms"]
            tpot = config["tpot_p50_ms"]
            latency = config["latency_p50_ms"]
            
            print(f"{batch:<8} {rps:<10.2f} {tok_s:<12.1f} {ttft:<12.1f} {tpot:<12.2f} {latency:<12.1f}")
            
            analysis["comparison"].append({
                "batch_size": batch,
                "throughput_rps": rps,
                "throughput_tok_s": tok_s,
                "ttft_p50_ms": ttft,
                "tpot_p50_ms": tpot,
                "latency_p50_ms": latency
            })
            
            if rps > max_throughput:
                max_throughput = rps
                best_throughput_config = batch
            
            if latency < min_latency:
                min_latency = latency
                best_latency_config = batch
        
        print("-"*70)
        
        analysis["summary"]["best_throughput"] = {
            "batch_size": best_throughput_config,
            "throughput_rps": max_throughput
        }
        analysis["summary"]["best_latency"] = {
            "batch_size": best_latency_config,
            "latency_p50_ms": min_latency
        }
        
        # Calculate improvement
        if len(configs) >= 2:
            baseline = configs[0]
            optimized = configs[-1]
            
            throughput_improvement = ((optimized["throughput_rps"] / baseline["throughput_rps"]) - 1) * 100
            latency_change = ((optimized["latency_p50_ms"] / baseline["latency_p50_ms"]) - 1) * 100
            
            analysis["improvement"] = {
                "baseline_batch": baseline["batch_size"],
                "optimized_batch": optimized["batch_size"],
                "throughput_improvement_pct": throughput_improvement,
                "latency_change_pct": latency_change
            }
            
            print(f"\n📊 Improvement Analysis:")
            print(f"   Baseline (batch={baseline['batch_size']}): {baseline['throughput_rps']:.2f} req/s")
            print(f"   Optimized (batch={optimized['batch_size']}): {optimized['throughput_rps']:.2f} req/s")
            print(f"   Improvement: {throughput_improvement:+.1f}%")
            
            if throughput_improvement > 10:
                analysis["summary"]["recommendation"] = f"Use batch_size={optimized['batch_size']} for {throughput_improvement:.0f}% throughput improvement"
            else:
                analysis["summary"]["recommendation"] = f"Batch size {baseline['batch_size']} is sufficient; larger batches show diminishing returns"
        
        return analysis
    
    def generate_report(self, results: Dict, analysis: Dict) -> Path:
        """Generate HTML comparison report."""
        report_file = self.output_dir / "batch_comparison_report.html"
        
        html = f'''<!DOCTYPE html>
<html>
<head>
    <title>Batch Size Optimization Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; background: #f5f6fa; }}
        .container {{ max-width: 1000px; margin: 0 auto; background: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
        h2 {{ color: #34495e; margin-top: 30px; }}
        .metric-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin: 20px 0; }}
        .metric-card {{ background: #f8f9fa; padding: 15px; border-radius: 5px; border-left: 4px solid #3498db; }}
        .metric-value {{ font-size: 24px; font-weight: bold; color: #2c3e50; }}
        .metric-label {{ font-size: 12px; color: #7f8c8d; text-transform: uppercase; }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background: #3498db; color: white; }}
        tr:hover {{ background: #f5f5f5; }}
        .best {{ background: #d5f4e6 !important; font-weight: bold; }}
        .improvement {{ color: #27ae60; font-weight: bold; }}
        .degradation {{ color: #e74c3c; font-weight: bold; }}
        .recommendation {{ background: #e8f6f3; padding: 20px; border-radius: 5px; border-left: 4px solid #1abc9c; margin: 20px 0; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>📊 Batch Size Optimization Report</h1>
        
        <div class="info-box">
            <p><strong>Model:</strong> {results.get("model", "N/A")}</p>
            <p><strong>Profile:</strong> {results.get("profile", "N/A")}</p>
            <p><strong>Date:</strong> {results.get("timestamp", "N/A")}</p>
        </div>
'''
        
        if analysis.get("summary", {}).get("recommendation"):
            html += f'''
        <div class="recommendation">
            <h3>💡 Recommendation</h3>
            <p>{analysis["summary"]["recommendation"]}</p>
        </div>
'''
        
        html += '''
        <h2>Performance Comparison</h2>
        <table>
            <tr>
                <th>Batch Size</th>
                <th>Throughput (req/s)</th>
                <th>Throughput (tok/s)</th>
                <th>TTFT p50 (ms)</th>
                <th>TPOT p50 (ms)</th>
                <th>Latency p50 (ms)</th>
            </tr>
'''
        
        best_throughput = analysis.get("summary", {}).get("best_throughput", {}).get("batch_size")
        best_latency = analysis.get("summary", {}).get("best_latency", {}).get("batch_size")
        
        for config in analysis.get("comparison", []):
            batch = config["batch_size"]
            row_class = ""
            if batch == best_throughput:
                row_class = "best"
            
            html += f'''
            <tr class="{row_class}">
                <td>{batch}</td>
                <td>{config["throughput_rps"]:.2f}</td>
                <td>{config["throughput_tok_s"]:.1f}</td>
                <td>{config["ttft_p50_ms"]:.1f}</td>
                <td>{config["tpot_p50_ms"]:.2f}</td>
                <td>{config["latency_p50_ms"]:.1f}</td>
            </tr>
'''
        
        html += '</table>'
        
        if analysis.get("improvement"):
            imp = analysis["improvement"]
            throughput_class = "improvement" if imp["throughput_improvement_pct"] > 0 else "degradation"
            latency_class = "improvement" if imp["latency_change_pct"] < 0 else "degradation"
            
            html += f'''
        <h2>Improvement Analysis</h2>
        <div class="metric-grid">
            <div class="metric-card">
                <div class="metric-value {throughput_class}">{imp["throughput_improvement_pct"]:+.1f}%</div>
                <div class="metric-label">Throughput Improvement</div>
            </div>
            <div class="metric-card">
                <div class="metric-value {latency_class}">{imp["latency_change_pct"]:+.1f}%</div>
                <div class="metric-label">Latency Change</div>
            </div>
        </div>
        
        <p><strong>Comparison:</strong> Batch {imp["baseline_batch"]} vs Batch {imp["optimized_batch"]}</p>
'''
        
        html += '''
        <h2>Key Insights</h2>
        <ul>
            <li>Larger batch sizes improve throughput by better utilizing GPU memory bandwidth</li>
            <li>TTFT (Time To First Token) may increase with larger batches due to queueing</li>
            <li>TPOT (Time Per Output Token) typically improves with batching</li>
            <li>Optimal batch size depends on latency requirements vs throughput needs</li>
        </ul>
        
        <div style="margin-top: 40px; padding-top: 20px; border-top: 1px solid #ddd; color: #7f8c8d; font-size: 12px;">
            Generated by Batch Size Optimization Tool
        </div>
    </div>
</body>
</html>
'''
        
        with open(report_file, 'w') as f:
            f.write(html)
        
        return report_file


def main():
    parser = argparse.ArgumentParser(description="Batch Size Optimization for vLLM")
    parser.add_argument("--model", default="TheBloke/Llama-2-7B-AWQ",
                       help="Model to test")
    parser.add_argument("--quantization", default="awq",
                       help="Quantization type")
    parser.add_argument("--profile", default="chat_medium",
                       help="Load profile")
    parser.add_argument("--duration", type=int, default=60,
                       help="Test duration per configuration")
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 4, 8, 16],
                       help="Batch sizes to test")
    parser.add_argument("--output-dir", default="varcas/profiles/roofline/batch_optimization",
                       help="Output directory")
    
    args = parser.parse_args()
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / f"comparison_{timestamp}"
    
    print("="*70)
    print("BATCH SIZE OPTIMIZATION FOR VLLM")
    print("="*70)
    print(f"Model: {args.model}")
    print(f"Profile: {args.profile}")
    print(f"Batch sizes to test: {args.batch_sizes}")
    print(f"Duration per config: {args.duration}s")
    print(f"Output: {output_dir}")
    print("="*70)
    
    # Run comparison
    optimizer = BatchSizeOptimizer(output_dir)
    
    results = optimizer.run_comparison(
        model=args.model,
        quantization=args.quantization,
        profile=args.profile,
        duration=args.duration,
        batch_sizes=args.batch_sizes
    )
    
    # Save raw results
    results_file = output_dir / "raw_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Analyze results
    analysis = optimizer.analyze_results(results)
    
    analysis_file = output_dir / "analysis.json"
    with open(analysis_file, 'w') as f:
        json.dump(analysis, f, indent=2)
    
    # Generate report
    report_file = optimizer.generate_report(results, analysis)
    
    # Final summary
    print("\n" + "="*70)
    print("OPTIMIZATION COMPLETE")
    print("="*70)
    print(f"\nResults saved to: {output_dir}")
    print(f"  - Raw results: {results_file}")
    print(f"  - Analysis: {analysis_file}")
    print(f"  - Report: {report_file}")
    
    if analysis.get("summary", {}).get("recommendation"):
        print(f"\n💡 {analysis['summary']['recommendation']}")
    
    print("\n" + "="*70)


if __name__ == "__main__":
    main()
