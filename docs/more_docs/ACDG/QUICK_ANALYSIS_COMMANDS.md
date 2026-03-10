# Quick Reference: Common Analysis Commands

Copy-paste these commands to analyze ACDG transformations.

## Analyze a Single Test Case

```bash
cd /scratch/general/vast/u1418973/mlir-air

# matmul_nd - producer-consumer pattern
./install/bin/air-opt mlir/test/Transform/AIRDependency/matmul_nd.mlir -air-dependency > /tmp/matmul.mlir
python3 analyze_acdg.py /tmp/matmul.mlir

# affine_if - branching pattern
./install/bin/air-opt mlir/test/Transform/AIRDependency/affine_if.mlir -air-dependency > /tmp/affine.mlir
python3 analyze_acdg.py /tmp/affine.mlir

# scf_for - loop pattern
./install/bin/air-opt mlir/test/Transform/AIRDependency/scf_for.mlir -air-dependency > /tmp/scf_for.mlir
python3 analyze_acdg.py /tmp/scf_for.mlir
```

## Compare Before & After

```bash
# Save original
./install/bin/air-opt mlir/test/Transform/AIRDependency/matmul_nd.mlir \
  --pass-pipeline="builtin.module()" > /tmp/before.mlir

# Save transformed
./install/bin/air-opt mlir/test/Transform/AIRDependency/matmul_nd.mlir \
  -air-dependency > /tmp/after.mlir

# Analyze both
echo "=== BEFORE ==="
python3 analyze_acdg.py /tmp/before.mlir

echo -e "\n=== AFTER ==="
python3 analyze_acdg.py /tmp/after.mlir
```

## Compare Raw vs Canonicalized

```bash
# Raw pass output
./install/bin/air-opt mlir/test/Transform/AIRDependency/matmul_nd.mlir \
  -air-dependency > /tmp/raw.mlir

# With canonicalization
./install/bin/air-opt mlir/test/Transform/AIRDependency/matmul_nd.mlir \
  --pass-pipeline="builtin.module(air-dependency,air-dependency-canonicalize)" \
  > /tmp/canonical.mlir

echo "Raw dependencies:"
python3 analyze_acdg.py /tmp/raw.mlir | grep -A 2 "TOKEN USAGE"

echo -e "\nCanonical dependencies:"
python3 analyze_acdg.py /tmp/canonical.mlir | grep -A 2 "TOKEN USAGE"
```

## Analyze Full Pipeline

```bash
# Run complete optimization pipeline
./install/bin/air-opt mlir/test/Transform/AIRDependency/matmul_nd.mlir \
  --pass-pipeline="builtin.module(air-dependency,canonicalize,cse,air-dependency-canonicalize,canonicalize,cse,air-dependency-schedule-opt)" \
  > /tmp/full_pipeline.mlir

# Analyze result
python3 analyze_acdg.py /tmp/full_pipeline.mlir
```

## Find Operations with Specific Patterns

```bash
# Find all operations that depend on multiple tokens (join points)
grep "depends on:.*,.*" /tmp/output.mlir

# Find all operations with no dependencies (roots)
grep -v "depends on" /tmp/output.mlir

# Find specific token references
grep "%async_token_4" /tmp/output.mlir

# Count operations by type
grep -o "air\.[a-z_]*" /tmp/output.mlir | sort | uniq -c

# See unused tokens
python3 analyze_acdg.py /tmp/output.mlir | grep -A 10 "Unused tokens"
```

## Extract Dependency Graph Details

```bash
# Show all dependencies in simple format
grep "async \[" /tmp/output.mlir | sed 's/.*async \[\(.*\)\].*/\1/'

# Count total dependencies
grep -o "async \[" /tmp/output.mlir | wc -l

# Find operations with empty dependencies
grep "async \[\]" /tmp/output.mlir

# Show operation IDs and their dependencies
grep "{id =" /tmp/output.mlir | grep -o "{id = [0-9]*}" | sort | uniq -c
```

## Analyze a Custom File

```bash
# Transform your own MLIR file
./install/bin/air-opt /path/to/your/file.mlir -air-dependency > /tmp/my_output.mlir

# Analyze it
python3 analyze_acdg.py /tmp/my_output.mlir

# With canonicalization
./install/bin/air-opt /path/to/your/file.mlir \
  --pass-pipeline="builtin.module(air-dependency,air-dependency-canonicalize)" \
  > /tmp/my_canonical.mlir

python3 analyze_acdg.py /tmp/my_canonical.mlir
```

## Run All Test Cases

```bash
# Process all test cases
for test in mlir/test/Transform/AIRDependency/*.mlir; do
  name=$(basename "$test" .mlir)
  echo "Processing $name..."
  ./install/bin/air-opt "$test" -air-dependency > "/tmp/${name}_async.mlir" 2>&1
done

# Analyze one test
python3 analyze_acdg.py /tmp/matmul_nd_async.mlir

# Compare all test sizes
for test in /tmp/*_async.mlir; do
  ops=$(python3 analyze_acdg.py "$test" | grep "Total async" | grep -o "[0-9]*" | head -1)
  name=$(basename "$test")
  echo "$name: $ops operations"
done
```

## Debugging Specific Issues

### Check if transformation worked
```bash
# Should see air.execute and async operations
grep -c "air.execute" /tmp/output.mlir
grep -c "async" /tmp/output.mlir
```

### Find bottlenecks
```bash
# Count join operations (multiple dependencies)
grep -c ", %" /tmp/output.mlir
```

### Check for cycles (shouldn't have any)
```bash
# Extract dependency graph
grep "async \[" /tmp/output.mlir | wc -l
# Should be fewer edges than operations
```

### Verify all tokens are used
```bash
# Compare token production to usage
echo "Produced:"
grep -o "air.execute\|air.dma_memcpy_nd\|air.wait_all" /tmp/output.mlir | wc -l
echo "Used (in async dependencies):"
grep "async \[" /tmp/output.mlir | wc -l
```

## Quick Workflow

```bash
# 1. Transform
./install/bin/air-opt test.mlir -air-dependency > /tmp/out.mlir

# 2. Quick check
python3 analyze_acdg.py /tmp/out.mlir | head -30

# 3. Look for issues
python3 analyze_acdg.py /tmp/out.mlir | tail -20

# 4. Compare if needed
diff <(python3 analyze_acdg.py /tmp/before.mlir) \
     <(python3 analyze_acdg.py /tmp/after.mlir)
```

## Save & Compare Results

```bash
# Archive results for later comparison
mkdir -p /tmp/acdg_results

for test in mlir/test/Transform/AIRDependency/*.mlir; do
  name=$(basename "$test" .mlir)
  ./install/bin/air-opt "$test" -air-dependency > /tmp/acdg_results/${name}.mlir 2>&1
  python3 analyze_acdg.py /tmp/acdg_results/${name}.mlir > /tmp/acdg_results/${name}_analysis.txt
done

# View all analyses
for f in /tmp/acdg_results/*_analysis.txt; do
  echo "=== $(basename $f) ==="
  head -20 "$f"
  echo ""
done
```

---

**Pro Tip:** Pipe analyze_acdg.py output to `less` for large files:
```bash
python3 analyze_acdg.py /tmp/output.mlir | less
```

**Pro Tip:** Create an alias for faster typing:
```bash
alias analyze='python3 /scratch/general/vast/u1418973/mlir-air/analyze_acdg.py'
analyze /tmp/output.mlir
```

**Pro Tip:** Combine with grep for targeted analysis:
```bash
python3 analyze_acdg.py /tmp/output.mlir | grep "Join operations" -A 10
```
