#!/usr/bin/env python3
"""Check if two result files have identical prompts and outputs."""

import json
import sys

def load_results(path):
    """Load results sorted by timestamp to preserve order."""
    data = json.load(open(path))
    records = data['records']
    # Sort by timestamp to get consistent ordering
    records.sort(key=lambda r: r.get('timestamp_sent', 0))
    return records

def compare(run1, run2):
    r1 = load_results(run1)
    r2 = load_results(run2)
    
    if len(r1) != len(r2):
        print(f"ERROR: Different number of requests! ({len(r1)} vs {len(r2)})")
        return
    
    print(f"Comparing {len(r1)} requests...")
    
    prompt_mismatches = 0
    output_mismatches = 0
    sample_mismatches = []
    
    for i, (rec1, rec2) in enumerate(zip(r1, r2)):
        if rec1['prompt_text'] != rec2['prompt_text']:
            prompt_mismatches += 1
        if rec1['output_text'] != rec2['output_text']:
            output_mismatches += 1
            if len(sample_mismatches) < 3:
                sample_mismatches.append({
                    'index': i,
                    'run1': rec1['output_text'][:100],
                    'run2': rec2['output_text'][:100]
                })
    
    print(f"\nResults:")
    print(f"  Prompt mismatches:  {prompt_mismatches}/{len(r1)}")
    print(f"  Output mismatches:  {output_mismatches}/{len(r1)}")
    
    if prompt_mismatches == 0 and output_mismatches == 0:
        print("\n✓ All prompts and outputs are IDENTICAL")
        print("  Variation is from timing jitter, not content")
    else:
        if prompt_mismatches > 0:
            print(f"\n⚠️  Prompts differ! Trace replay may not be working correctly.")
        if output_mismatches > 0:
            print(f"\n⚠️  {output_mismatches} outputs differ")
            print("  vLLM is non-deterministic even with temperature=0")
            print("\n  Sample differences:")
            for m in sample_mismatches:
                print(f"\n  Request #{m['index']}:")
                print(f"    Run1: {m['run1']}...")
                print(f"    Run2: {m['run2']}...")

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} run1.json run2.json")
        sys.exit(1)
    compare(sys.argv[1], sys.argv[2])
