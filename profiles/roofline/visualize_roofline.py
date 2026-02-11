#!/usr/bin/env python3
"""
Roofline Plot Visualization

Generates publication-quality roofline plots from static and dynamic analysis results.

Usage:
    python visualize_roofline.py --static static_analysis.json --dynamic dynamic_analysis.json
    python visualize_roofline.py --static static_analysis.json
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional
import math


def create_roofline_svg(roofline_data: Dict, output_file: Path, 
                        title: str = "Roofline Analysis",
                        width: int = 800, height: int = 600) -> str:
    """
    Create an SVG roofline plot.
    
    The roofline plot shows:
    - X-axis: Arithmetic Intensity (FLOP/Byte) - log scale
    - Y-axis: Performance (TFLOPS) - log scale
    - Roofline: Theoretical peak performance bound
    - Points: Measured or estimated kernel/workload performance
    """
    
    # Margins
    margin_left = 80
    margin_right = 40
    margin_top = 60
    margin_bottom = 80
    
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom
    
    # Data ranges
    min_ai = 0.1
    max_ai = 1000
    min_tflops = 0.1
    max_tflops = roofline_data.get("peak_tflops", 100) * 1.2
    
    def ai_to_x(ai: float) -> float:
        """Convert AI to X coordinate."""
        log_min = math.log10(min_ai)
        log_max = math.log10(max_ai)
        log_ai = math.log10(max(min_ai, min(max_ai, ai)))
        return margin_left + (log_ai - log_min) / (log_max - log_min) * plot_width
    
    def tflops_to_y(tflops: float) -> float:
        """Convert TFLOPS to Y coordinate."""
        log_min = math.log10(min_tflops)
        log_max = math.log10(max_tflops)
        log_tflops = math.log10(max(min_tflops, min(max_tflops, tflops)))
        return height - margin_bottom - (log_tflops - log_min) / (log_max - log_min) * plot_height
    
    svg_parts = []
    
    # Header
    svg_parts.append(f'''<?xml version="1.0" encoding="UTF-8"?>
<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">
<defs>
    <style>
        .title {{ font-family: Arial, sans-serif; font-size: 18px; font-weight: bold; }}
        .axis-label {{ font-family: Arial, sans-serif; font-size: 12px; }}
        .tick-label {{ font-family: Arial, sans-serif; font-size: 10px; }}
        .legend {{ font-family: Arial, sans-serif; font-size: 11px; }}
        .roofline {{ fill: none; stroke: #e74c3c; stroke-width: 3; }}
        .memory-slope {{ fill: none; stroke: #3498db; stroke-width: 2; stroke-dasharray: 5,5; }}
        .point-gemm {{ fill: #e74c3c; stroke: #c0392b; stroke-width: 1; }}
        .point-attention {{ fill: #9b59b6; stroke: #8e44ad; stroke-width: 1; }}
        .point-memory {{ fill: #3498db; stroke: #2980b9; stroke-width: 1; }}
        .point-workload {{ fill: #2ecc71; stroke: #27ae60; stroke-width: 2; }}
        .grid-line {{ stroke: #ecf0f1; stroke-width: 1; }}
        .axis-line {{ stroke: #2c3e50; stroke-width: 2; }}
    </style>
</defs>
''')
    
    # Background
    svg_parts.append(f'<rect width="{width}" height="{height}" fill="#ffffff"/>')
    
    # Title
    svg_parts.append(f'<text x="{width/2}" y="{margin_top - 20}" text-anchor="middle" class="title">{title}</text>')
    
    # Grid lines
    for ai in [0.1, 1, 10, 100, 1000]:
        x = ai_to_x(ai)
        svg_parts.append(f'<line x1="{x}" y1="{margin_top}" x2="{x}" y2="{height - margin_bottom}" class="grid-line"/>')
    
    for tflops in [0.1, 1, 10, 100]:
        y = tflops_to_y(tflops)
        svg_parts.append(f'<line x1="{margin_left}" y1="{y}" x2="{width - margin_right}" y2="{y}" class="grid-line"/>')
    
    # Axes
    svg_parts.append(f'<line x1="{margin_left}" y1="{height - margin_bottom}" x2="{width - margin_right}" y2="{height - margin_bottom}" class="axis-line"/>')
    svg_parts.append(f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{height - margin_bottom}" class="axis-line"/>')
    
    # Axis labels
    svg_parts.append(f'<text x="{width/2}" y="{height - 20}" text-anchor="middle" class="axis-label">Arithmetic Intensity (FLOP/Byte)</text>')
    svg_parts.append(f'<text x="{20}" y="{height/2}" text-anchor="middle" transform="rotate(-90, 20, {height/2})" class="axis-label">Performance (TFLOPS)</text>')
    
    # Tick labels
    for ai, label in [(0.1, "0.1"), (1, "1"), (10, "10"), (100, "100"), (1000, "1000")]:
        x = ai_to_x(ai)
        svg_parts.append(f'<text x="{x}" y="{height - margin_bottom + 20}" text-anchor="middle" class="tick-label">{label}</text>')
    
    for tflops, label in [(0.1, "0.1"), (1, "1"), (10, "10"), (100, "100")]:
        if tflops <= max_tflops:
            y = tflops_to_y(tflops)
            svg_parts.append(f'<text x="{margin_left - 10}" y="{y + 4}" text-anchor="end" class="tick-label">{label}</text>')
    
    # Roofline curve
    peak_tflops = roofline_data.get("peak_tflops", 100)
    mem_bw = roofline_data.get("memory_bw_gbps", 1000)
    ridge_point = peak_tflops / (mem_bw / 1000)  # Convert GB/s to TB/s for matching units
    
    # Memory-bound line (left of ridge point)
    memory_points = []
    for ai in [min_ai, ridge_point / 2, ridge_point * 0.95]:
        tflops = ai * mem_bw / 1000  # AI * BW (TB/s) = TFLOPS
        memory_points.append((ai_to_x(ai), tflops_to_y(tflops)))
    
    # Compute-bound line (right of ridge point)
    compute_points = []
    for ai in [ridge_point, max_ai]:
        tflops = peak_tflops
        compute_points.append((ai_to_x(ai), tflops_to_y(tflops)))
    
    # Draw roofline
    if len(memory_points) >= 2:
        path = f"M {memory_points[0][0]} {memory_points[0][1]}"
        for p in memory_points[1:]:
            path += f" L {p[0]} {p[1]}"
        for p in compute_points:
            path += f" L {p[0]} {p[1]}"
        svg_parts.append(f'<path d="{path}" class="roofline"/>')
    
    # Ridge point marker
    ridge_x = ai_to_x(ridge_point)
    ridge_y = tflops_to_y(peak_tflops)
    svg_parts.append(f'<circle cx="{ridge_x}" cy="{ridge_y}" r="5" fill="#e74c3c"/>')
    svg_parts.append(f'<text x="{ridge_x + 10}" y="{ridge_y - 10}" class="legend">Ridge Point ({ridge_point:.1f})</text>')
    
    # Memory bandwidth annotation
    mem_label_x = ai_to_x(min_ai * 3)
    mem_label_y = tflops_to_y(min_ai * 3 * mem_bw / 1000)
    svg_parts.append(f'<text x="{mem_label_x}" y="{mem_label_y - 10}" class="legend" fill="#3498db">Memory Bandwidth: {mem_bw} GB/s</text>')
    
    # Peak performance annotation
    svg_parts.append(f'<text x="{width - margin_right - 10}" y="{ridge_y - 10}" text-anchor="end" class="legend" fill="#e74c3c">Peak: {peak_tflops} TFLOPS</text>')
    
    # Plot measured points
    points = roofline_data.get("measured_points", [])
    workload_points = roofline_data.get("workload_points", [])
    
    all_points = []
    
    # Categorize points
    for p in points + workload_points:
        name = p.get("name", "").lower()
        ai = p.get("ai", 0)
        tflops = p.get("tflops", 0)
        
        if ai > 0 and tflops > 0:
            point_class = "point-memory"
            if "gemm" in name or "matmul" in name or "gemv" in name:
                point_class = "point-gemm"
            elif "attention" in name or "flash" in name or "softmax" in name:
                point_class = "point-attention"
            elif "workload" in name or "prefill" in name or "decode" in name:
                point_class = "point-workload"
            
            all_points.append({
                "x": ai_to_x(ai),
                "y": tflops_to_y(tflops),
                "class": point_class,
                "name": p.get("name", "")[:30],
                "ai": ai,
                "tflops": tflops
            })
    
    # Draw points
    for p in all_points:
        svg_parts.append(f'<circle cx="{p["x"]}" cy="{p["y"]}" r="4" class="{p["class"]}"/>')
    
    # Legend
    legend_x = width - margin_right - 150
    legend_y = margin_top + 20
    
    svg_parts.append(f'<rect x="{legend_x - 10}" y="{legend_y - 15}" width="140" height="80" fill="white" stroke="#bdc3c7"/>')
    svg_parts.append(f'<text x="{legend_x}" y="{legend_y}" class="legend" font-weight="bold">Legend</text>')
    
    svg_parts.append(f'<circle cx="{legend_x + 5}" cy="{legend_y + 20}" r="4" class="point-gemm"/>')
    svg_parts.append(f'<text x="{legend_x + 15}" y="{legend_y + 24}" class="legend">GEMM kernels</text>')
    
    svg_parts.append(f'<circle cx="{legend_x + 5}" cy="{legend_y + 40}" r="4" class="point-attention"/>')
    svg_parts.append(f'<text x="{legend_x + 15}" y="{legend_y + 44}" class="legend">Attention</text>')
    
    svg_parts.append(f'<circle cx="{legend_x + 5}" cy="{legend_y + 60}" r="4" class="point-workload"/>')
    svg_parts.append(f'<text x="{legend_x + 15}" y="{legend_y + 64}" class="legend">Workloads</text>')
    
    # Close SVG
    svg_parts.append('</svg>')
    
    return '\n'.join(svg_parts)


def create_comparison_html(static_data: Dict, dynamic_data: Optional[Dict], 
                          output_file: Path):
    """Create an HTML report comparing static and dynamic roofline."""
    
    html = '''<!DOCTYPE html>
<html>
<head>
    <title>Roofline Analysis Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; background: #f5f6fa; }
        .container { max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        h1 { color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }
        h2 { color: #34495e; margin-top: 30px; }
        .info-box { background: #ecf0f1; padding: 15px; border-radius: 5px; margin: 20px 0; }
        .metric-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin: 20px 0; }
        .metric-card { background: #f8f9fa; padding: 15px; border-radius: 5px; border-left: 4px solid #3498db; }
        .metric-value { font-size: 24px; font-weight: bold; color: #2c3e50; }
        .metric-label { font-size: 12px; color: #7f8c8d; text-transform: uppercase; }
        .plot-container { margin: 30px 0; text-align: center; }
        .plot-container svg { max-width: 100%; height: auto; border: 1px solid #ddd; border-radius: 5px; }
        table { width: 100%; border-collapse: collapse; margin: 20px 0; }
        th, td { padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }
        th { background: #3498db; color: white; }
        tr:hover { background: #f5f5f5; }
        .code { font-family: monospace; background: #f4f4f4; padding: 2px 5px; border-radius: 3px; }
        .bound-compute { color: #e74c3c; font-weight: bold; }
        .bound-memory { color: #3498db; font-weight: bold; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🚀 Roofline Analysis Report</h1>
'''
    
    # Static analysis section
    if static_data:
        gpu = static_data.get("gpu", {})
        model = static_data.get("model", {})
        
        html += f'''
        <div class="info-box">
            <h3>Configuration</h3>
            <p><strong>GPU:</strong> {gpu.get("name", "Unknown")} | 
               <strong>Model:</strong> {model.get("name", "Unknown")} | 
               <strong>Parameters:</strong> {model.get("params", "?")}B ({model.get("bits", 16)}-bit)</p>
        </div>
        
        <div class="metric-grid">
            <div class="metric-card">
                <div class="metric-value">{gpu.get("peak_fp16_flops", 0):.1f}</div>
                <div class="metric-label">Peak FP16 (TFLOPS)</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{gpu.get("memory_bw", 0):.0f}</div>
                <div class="metric-label">Memory BW (GB/s)</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{gpu.get("peak_fp16_flops", 0) / (gpu.get("memory_bw", 1) / 1000):.1f}</div>
                <div class="metric-label">Ridge Point</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{model.get("params", 0) * model.get("bits", 16) / 16:.1f}B</div>
                <div class="metric-label">Effective Params</div>
            </div>
        </div>
'''
        
        # Static roofline plot
        if "roofline_plot_data" in static_data:
            html += '''
        <h2>📊 Static Roofline Analysis</h2>
        <p>Theoretical roofline based on model architecture and GPU specifications.</p>
        <div class="plot-container">
'''
            svg = create_roofline_svg(
                static_data["roofline_plot_data"], 
                output_file.parent / "static_roofline.svg",
                "Static Roofline - Theoretical Bounds"
            )
            html += svg
            html += '</div>'
        
        # Prefill/Decode analysis table
        html += '''
        <h2>🔍 Phase Analysis</h2>
        <table>
            <tr>
                <th>Phase</th>
                <th>Batch</th>
                <th>Seq Len</th>
                <th>Arithmetic Intensity</th>
                <th>Attainable TFLOPS</th>
                <th>Utilization</th>
                <th>Bound</th>
            </tr>
'''
        
        for result in static_data.get("prefill_results", [])[:4]:
            bound_class = "bound-compute" if result.get("compute_utilization_pct", 0) > 80 else "bound-memory"
            bound_text = "COMPUTE" if result.get("compute_utilization_pct", 0) > 80 else "MEMORY"
            html += f'''
            <tr>
                <td class="code">Prefill</td>
                <td>{result.get("batch_size", "?")}</td>
                <td>{result.get("seq_len", "?")}</td>
                <td>{result.get("arithmetic_intensity", 0):.2f}</td>
                <td>{result.get("attainable_tflops", 0):.2f}</td>
                <td>{result.get("compute_utilization_pct", 0):.1f}%</td>
                <td class="{bound_class}">{bound_text}</td>
            </tr>
'''
        
        for result in static_data.get("decode_results", [])[:4]:
            bound_class = "bound-compute" if result.get("compute_utilization_pct", 0) > 80 else "bound-memory"
            bound_text = "COMPUTE" if result.get("compute_utilization_pct", 0) > 80 else "MEMORY"
            html += f'''
            <tr>
                <td class="code">Decode</td>
                <td>{result.get("batch_size", "?")}</td>
                <td>{result.get("context_len", "?")}</td>
                <td>{result.get("arithmetic_intensity", 0):.2f}</td>
                <td>{result.get("attainable_tflops", 0):.2f}</td>
                <td>{result.get("compute_utilization_pct", 0):.1f}%</td>
                <td class="{bound_class}">{bound_text}</td>
            </tr>
'''
        
        html += '</table>'
    
    # Dynamic analysis section
    if dynamic_data:
        html += '''
        <h2>📈 Dynamic Roofline Analysis</h2>
        <p>Measured roofline from actual GPU profiling.</p>
'''
        
        if "roofline" in dynamic_data:
            html += '<div class="plot-container">'
            svg = create_roofline_svg(
                dynamic_data["roofline"],
                output_file.parent / "dynamic_roofline.svg",
                "Dynamic Roofline - Measured Performance"
            )
            html += svg
            html += '</div>'
        
        if "kernel_analysis" in dynamic_data:
            ka = dynamic_data["kernel_analysis"]
            summary = ka.get("summary", {})
            
            html += f'''
        <div class="metric-grid">
            <div class="metric-card">
                <div class="metric-value">{summary.get("overall_ai", 0):.2f}</div>
                <div class="metric-label">Measured AI</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{summary.get("overall_tflops", 0):.2f}</div>
                <div class="metric-label">Achieved TFLOPS</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{summary.get("total_kernels", 0)}</div>
                <div class="metric-label">Kernels Profiled</div>
            </div>
        </div>
        
        <h3>Top Kernels by Duration</h3>
        <table>
            <tr>
                <th>Kernel</th>
                <th>Duration (ms)</th>
                <th>AI</th>
                <th>TFLOPS</th>
            </tr>
'''
            for kernel in ka.get("top_kernels", [])[:10]:
                html += f'''
            <tr>
                <td class="code">{kernel.get("name", "unknown")[:50]}</td>
                <td>{kernel.get("duration_ms", 0):.3f}</td>
                <td>{kernel.get("ai", 0):.2f}</td>
                <td>{kernel.get("tflops", 0):.2f}</td>
            </tr>
'''
            html += '</table>'
    
    # Footer
    html += '''
        <div style="margin-top: 40px; padding-top: 20px; border-top: 1px solid #ddd; color: #7f8c8d; font-size: 12px;">
            Generated by Roofline Analysis Tool
        </div>
    </div>
</body>
</html>
'''
    
    with open(output_file, 'w') as f:
        f.write(html)
    
    return html


def main():
    parser = argparse.ArgumentParser(description="Visualize Roofline Analysis")
    parser.add_argument("--static", type=str, help="Static analysis JSON file")
    parser.add_argument("--dynamic", type=str, help="Dynamic analysis JSON file")
    parser.add_argument("--output", type=str, default="varcas/profiles/roofline/roofline_report.html",
                       help="Output HTML file")
    
    args = parser.parse_args()
    
    # Load data
    static_data = None
    dynamic_data = None
    
    if args.static:
        static_path = Path(args.static)
        if static_path.exists():
            with open(static_path) as f:
                static_data = json.load(f)
            print(f"Loaded static analysis from {static_path}")
        else:
            print(f"Static analysis file not found: {static_path}")
    
    if args.dynamic:
        dynamic_path = Path(args.dynamic)
        if dynamic_path.exists():
            with open(dynamic_path) as f:
                dynamic_data = json.load(f)
            print(f"Loaded dynamic analysis from {dynamic_path}")
        else:
            print(f"Dynamic analysis file not found: {dynamic_path}")
    
    if not static_data and not dynamic_data:
        print("No data to visualize. Please provide --static and/or --dynamic files.")
        return
    
    # Create report
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    create_comparison_html(static_data, dynamic_data, output_path)
    
    print(f"\nReport saved to: {output_path}")
    print(f"Open this file in a browser to view the interactive roofline plots.")


if __name__ == "__main__":
    main()
