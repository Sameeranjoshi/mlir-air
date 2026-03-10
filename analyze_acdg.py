#!/usr/bin/env python3
"""
ACDG Dependency Analyzer
Extracts and visualizes token dependencies from AIRDependency pass output.

Usage:
  python3 analyze_acdg.py <output.mlir>

This script helps you understand the dependency graph created by the
AIRDependency pass by extracting all async operations and their dependencies.
"""

import re
import sys
from collections import defaultdict

def extract_async_ops(mlir_file):
    """Extract all async operations (execute, dma, wait_all, herd, etc)."""
    with open(mlir_file, 'r') as f:
        content = f.read()

    ops = []

    # Pattern to match async operations with dependencies
    # Matches: %token[, %result] = operation async [deps] ... {id = N}
    pattern = r'(%\w+)(?:,\s*(%\w+))?\s*=\s*(air\.\w+|scf\.\w+|affine\.\w+)\s*(?:async\s*\[([^\]]*)\])?.*?\{id\s*=\s*(\d+)'

    for match in re.finditer(pattern, content):
        token, result, op_type, deps_str, op_id = match.groups()
        deps = [d.strip() for d in deps_str.split(',')] if deps_str and deps_str.strip() else []

        ops.append({
            'token': token.strip(),
            'result': result.strip() if result else None,
            'type': op_type,
            'deps': deps,
            'id': op_id
        })

    return ops

def categorize_ops(ops):
    """Categorize operations by type."""
    categories = defaultdict(list)
    for op in ops:
        categories[op['type']].append(op)
    return categories

def print_report(mlir_file):
    """Print detailed analysis report."""
    ops = extract_async_ops(mlir_file)
    categories = categorize_ops(ops)

    print("\n" + "="*100)
    print(f"ACDG DEPENDENCY ANALYSIS: {mlir_file}")
    print("="*100 + "\n")

    print(f"Total async operations: {len(ops)}\n")

    # Show operation counts by type
    print("Operation Breakdown:")
    for op_type, ops_list in sorted(categories.items()):
        print(f"  {op_type:30} {len(ops_list):3} operations")

    print("\n" + "-"*100)
    print("DETAILED OPERATION LIST:")
    print("-"*100 + "\n")
    print(f"{'Type':<30} {'Token':<20} {'ID':<4} {'Dependencies':<30}")
    print("-"*100)

    for op in ops:
        deps = ', '.join(op['deps']) if op['deps'] else '[none]'
        if len(deps) > 28:
            deps = deps[:25] + "..."
        print(f"{op['type']:<30} {op['token']:<20} {op['id']:<4} {deps:<30}")

    # Analyze dependency patterns
    print("\n" + "-"*100)
    print("DEPENDENCY PATTERNS:")
    print("-"*100 + "\n")

    # Find root operations (no deps)
    roots = [op for op in ops if not op['deps']]
    print(f"Root operations (no dependencies): {len(roots)}")
    for root in roots[:5]:
        print(f"  {root['token']:20} = {root['type']:<30} (id={root['id']})")

    if len(roots) > 5:
        print(f"  ... and {len(roots) - 5} more")

    # Find operations that depend on multiple tokens (joins)
    multi_deps = [op for op in ops if len(op['deps']) > 1]
    print(f"\nJoin operations (multiple dependencies): {len(multi_deps)}")
    for join in multi_deps[:5]:
        deps_str = ', '.join(join['deps'])
        print(f"  {join['token']:20} depends on: {deps_str}")

    if len(multi_deps) > 5:
        print(f"  ... and {len(multi_deps) - 5} more")

    # Analyze token usage
    print("\n" + "-"*100)
    print("TOKEN USAGE ANALYSIS:")
    print("-"*100 + "\n")

    # Track which tokens are used as dependencies
    all_deps = set()
    for op in ops:
        all_deps.update(op['deps'])

    produced_tokens = {op['token'] for op in ops}
    used_tokens = all_deps & produced_tokens
    unused_tokens = produced_tokens - all_deps

    print(f"Total tokens produced: {len(produced_tokens)}")
    print(f"Tokens actually used:  {len(used_tokens)}")
    print(f"Unused tokens:         {len(unused_tokens)}")

    if unused_tokens:
        print(f"\nUnused tokens:")
        for token in sorted(unused_tokens)[:5]:
            print(f"  {token}")
        if len(unused_tokens) > 5:
            print(f"  ... and {len(unused_tokens) - 5} more")

    print("\n" + "="*100 + "\n")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python3 analyze_acdg.py <mlir_file>")
        print("\nExample:")
        print("  ./install/bin/air-opt matmul_nd.mlir -air-dependency > /tmp/output.mlir")
        print("  python3 analyze_acdg.py /tmp/output.mlir")
        sys.exit(1)

    mlir_file = sys.argv[1]
    print_report(mlir_file)
