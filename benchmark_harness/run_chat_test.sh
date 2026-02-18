#!/bin/bash
#
# Chat Test Script for SLA Verification
# Matches the "concurrent users" parameter used in config_iq evaluation
#
# This script runs a closed-loop chat workload test that directly corresponds
# to the config_iq tool's "concurrent users" parameter for validation.
#

set -e

# Default values
VLLM_URL="http://localhost:8000"
CONCURRENT_USERS=1
DURATION=60
OUTPUT_DIR="./results"
MEANINGFUL_PROMPTS=""
TEMPERATURE=0.0

# Parse arguments
usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Run a chat workload test matching config_iq's 'concurrent users' parameter

Options:
  --url URL                vLLM endpoint URL (default: http://localhost:8000)
  --users N                Number of concurrent users (default: 100)
  --duration SECONDS       Test duration in seconds (default: 120)
  --output-dir DIR         Output directory for results (default: ./results)
  --meaningful-prompts     Use meaningful prompts instead of synthetic
  --temperature FLOAT      Sampling temperature (default: 0.0)
  --help                   Show this help message

Examples:
  $0 --users 50 --duration 60
  $0 --url http://vllm-server:8000 --users 200 --duration 180

SLA Targets (Chat Workload):
  TTFT P50: ≤200ms, TTFT P95: ≤500ms, TTFT P99: ≤1000ms
  TPOT P50: ≤50ms, TPOT P95: ≤80ms, TPOT P99: ≤120ms

Token Distribution (matches config_iq):
  Input:  mean=50, std=30, min=10, max=500
  Output: mean=150, std=80, min=20, max=1000
EOF
    exit 1
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --url)
            VLLM_URL="$2"
            shift 2
            ;;
        --users)
            CONCURRENT_USERS="$2"
            shift 2
            ;;
        --duration)
            DURATION="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --meaningful-prompts)
            MEANINGFUL_PROMPTS="--meaningful-prompts"
            shift
            ;;
        --temperature)
            TEMPERATURE="$2"
            shift 2
            ;;
        --help)
            usage
            ;;
        *)
            echo "Unknown option: $1"
            usage
            ;;
    esac
done

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Generate output filename with timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_FILE="$OUTPUT_DIR/chat_test_users${CONCURRENT_USERS}_${TIMESTAMP}.json"

cat << EOF
==========================================
Chat Test - SLA Verification
==========================================
vLLM URL:        $VLLM_URL
Concurrent Users: $CONCURRENT_USERS
Duration:        ${DURATION}s
Output File:     $OUTPUT_FILE
Temperature:     $TEMPERATURE
$(if [[ -n "$MEANINGFUL_PROMPTS" ]]; then echo "Prompts:         Meaningful"; else echo "Prompts:         Synthetic"; fi)
==========================================

SLA Targets (Chat Workload):
  TTFT P50: ≤200ms, TTFT P95: ≤500ms, TTFT P99: ≤1000ms
  TPOT P50: ≤50ms, TPOT P95: ≤80ms, TPOT P99: ≤120ms

Token Distribution:
  Input:  mean=50, std=30, min=10, max=500
  Output: mean=150, std=80, min=20, max=1000

EOF

# Check if vLLM is accessible
echo "Checking vLLM server..."
if ! curl -s "$VLLM_URL/health" > /dev/null 2>&1; then
    echo "Warning: Could not connect to vLLM at $VLLM_URL"
    echo "Make sure vLLM is running before starting the test."
    echo ""
    read -p "Continue anyway? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Run the test using the Python harness
echo ""
echo "Starting load test with $CONCURRENT_USERS concurrent users..."
echo ""

python3 "$SCRIPT_DIR/varcas_load_harness.py" \
    --url "$VLLM_URL" \
    --profile "closed_loop" \
    --duration "$DURATION" \
    --output "$OUTPUT_FILE" \
    --temperature "$TEMPERATURE" \
    $MEANINGFUL_PROMPTS \
    --concurrent-users "$CONCURRENT_USERS"

echo ""
echo "=========================================="
echo "Test Complete"
echo "=========================================="
echo "Results saved to: $OUTPUT_FILE"
echo ""

# Display summary if jq is available
if command -v jq &> /dev/null; then
    echo "Summary:"
    echo "--------"
    jq -r '.metrics | "Total Requests: \(.total_requests)
Success Rate: \((.successful_requests / .total_requests * 100) | floor)%
Throughput: \(.throughput_rps | floor) req/s, \(.throughput_tok_s | floor) tok/s
TTFT: p50=\(.ttft_p50_ms)ms, p99=\(.ttft_p99_ms)ms
TPOT: p50=\(.tpot_p50_ms)ms, p99=\(.tpot_p99_ms)ms
Latency: p50=\(.latency_p50_ms)ms, p99=\(.latency_p99_ms)ms"' "$OUTPUT_FILE"
    
    echo ""
    echo "SLA Targets for Chat Workload (from config_iq):"
    echo "------------------------------------------------"
    echo "TTFT P50: ≤200ms, TTFT P95: ≤500ms, TTFT P99: ≤1000ms"
    echo "TPOT P50: ≤50ms,  TPOT P95: ≤80ms,  TPOT P99: ≤120ms"
    
    echo ""
    echo "SLA Compliance Check:"
    echo "---------------------"
    
    # Function to check SLA and print result
    check_sla() {
        local metric="$1"
        local value="$2"
        local target="$3"
        
        # Use awk for float comparison
        if awk "BEGIN {exit !($value <= $target)}"; then
            printf "✓ %-12s: %6.1fms (target: ≤%dms)\n" "$metric" "$value" "$target"
        else
            local excess=$(awk "BEGIN {print $value - $target}")
            printf "✗ %-12s: %6.1fms (target: ≤%dms) - EXCEEDED by %.1fms\n" "$metric" "$value" "$target" "$excess"
        fi
    }
    
    # Extract metrics and check against targets
    TTFT_P50=$(jq -r '.metrics.ttft_p50_ms' "$OUTPUT_FILE")
    TTFT_P99=$(jq -r '.metrics.ttft_p99_ms' "$OUTPUT_FILE")
    TPOT_P50=$(jq -r '.metrics.tpot_p50_ms' "$OUTPUT_FILE")
    TPOT_P99=$(jq -r '.metrics.tpot_p99_ms' "$OUTPUT_FILE")
    
    check_sla "TTFT P50" "$TTFT_P50" 200
    check_sla "TTFT P99" "$TTFT_P99" 1000
    check_sla "TPOT P50" "$TPOT_P50" 50
    check_sla "TPOT P99" "$TPOT_P99" 120
else
    echo "Note: Install 'jq' for detailed SLA compliance report"
fi

echo ""
echo "Full results available at: $OUTPUT_FILE"
