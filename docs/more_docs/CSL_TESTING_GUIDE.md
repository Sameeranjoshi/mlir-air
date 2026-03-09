# CSL Dialect Testing Guide

## Overview

This guide explains how to test the clean two-level abstraction for CSL dialects:

- **CSL Dialect**: PE-level IR (kernels, functions, spatial placement)
- **CSL Runtime Dialect**: High-level SDK API (layout creation, parameter setting, compilation)
- **Pipeline**: `CSL → CSL Runtime Conversion → Python Emission`

---

## Architecture

```
┌──────────────────────────────────┐
│  CSL Dialect (PE-level IR)       │
│  - kernel, func, task, var       │
│  - spatial_placement, code_region│
│  - paint, port, dataflow         │
└──────────────┬───────────────────┘
               │ -csl-to-csl-rt conversion pass
               ↓
┌──────────────────────────────────┐
│  CSL Runtime Dialect (SDK API)   │
│  - create_layout                 │
│  - create_code_region            │
│  - place, set_param_all, compile │
└──────────────┬───────────────────┘
               │ --emit-csl-rt translation
               ↓
        Python Output (run.py)
  platform → layout → compile_artifacts → SdkRuntime
```

---

## Quick Start: Full Pipeline

### Layout and runtime integration

The emitted **run.py** uses the layout (from csl_rt) inside the same script:

1. **Platform:** `platform = get_platform(args.cmaddr, config, target)`
2. **Layout:** `layout = SdkLayout(platform)` then code regions, place, set_param_all, export_name (from csl_rt ops).
3. **Compile:** `compile_artifacts = layout.compile(out_prefix='out')`
4. **Runtime:** `runtime = SdkRuntime(compile_artifacts, platform, memcpy_required=False)`
5. **Run:** `runtime.load()`, `runtime.run()`, `runtime.stop()`

You do not pass a pre-compiled directory to `SdkRuntime`; the script builds the layout, compiles it, and creates the runtime from `compile_artifacts`.

### Single Command
```bash
# Convert CSL → CSL Runtime → Python (all in one)
air-opt input.mlir -csl-to-csl-rt | air-translate --emit-csl-rt -o output.py
```

### Step-by-Step

```bash
# Step 1: Parse CSL dialect
air-opt program.mlir

# Step 2: Convert to CSL Runtime
air-opt program.mlir -csl-to-csl-rt -o runtime.mlir

# Step 3: Emit to Python
air-opt program.mlir -csl-to-csl-rt | air-translate --emit-csl-rt -o output.py
```

---

## Test Cases

### Test 1: CSL Dialect Parsing

**Verify CSL PE-level ops parse correctly**

```bash
air-opt mlir/test/Dialect/CSL/kernel_ops.mlir
```

**Expected output**: CSL IR with `csl.var`, `csl.func`, `csl.task`, etc.

### Test 2: CSL→CSL Runtime Conversion

**Verify spatial_placement converts to CSL Runtime ops**

```bash
air-opt mlir/test/Conversion/CSLToCSLRuntime/layout_to_runtime.mlir -csl-to-csl-rt
```

**Expected output**:
- Input: `csl.spatial_placement`, `csl.code_region`, `csl.place`
- Output: `csl_rt.create_layout`, `csl_rt.place`, `csl_rt.compile`

### Test 3: Full Pipeline with Real Program

**Create a test CSL program**:

```mlir
%kernel = csl.kernel "compute.csl" params({tile_id = 0 : i32}) {
  csl.var @data : memref<512xf32>

  csl.func @process() : () -> () {
    csl.return
  }

  csl.comptime {
    csl.export_symbol @data alias("input_buffer")
    csl.export_symbol @process
  }
} : !csl.kernel

csl.spatial_placement {
  %color = csl.color : !csl.color
  %route = csl.route in(RAMP) out(WEST) : i32

  %code_region = csl.code_region routes(%route) colors(%color) shape(8, 8) {
    csl.paint pe(0, 0) route(%route) color(%color)
  } : !csl.code_region

  csl.place %code_region at(0, 0) kernel(%kernel)
}
```

**Run the pipeline**:

```bash
air-opt test.mlir -csl-to-csl-rt | air-translate --emit-csl-rt -o output.py
```

---

## Validation Checklist

### ✅ CSL Parsing
- [ ] `air-opt mlir/test/Dialect/CSL/kernel_ops.mlir` succeeds
- [ ] `air-opt mlir/test/Dialect/CSL/roundtrip.mlir --verify-roundtrip` succeeds
- [ ] CSL ops display correctly (csl.kernel, csl.func, csl.task, etc.)

### ✅ CSL Runtime Conversion
- [ ] `air-opt -csl-to-csl-rt` produces CSL Runtime ops (csl_rt.*)
- [ ] Spatial placement ops convert to layout ops
- [ ] PE-level kernel ops remain unchanged

### ✅ Translation
- [ ] `air-translate --emit-csl-rt` produces Python output
- [ ] No `--emit-csl` (direct CSL→Text is removed)
- [ ] Both dialects are registered in translator

### ✅ Architecture
- [ ] No direct CSL→Text path (clean separation)
- [ ] Conversion pass is mandatory for layout ops
- [ ] Single translation backend (--emit-csl-rt) for output

---

## Running CSL tests (ninja)

From the build directory:

```bash
# Run all CSL-related tests (dialect, CSL Runtime dialect, conversions, emit)
ninja check-csl-all
```

Individual suites:

| Target | Tests |
|--------|--------|
| `ninja check-csl` | CSL dialect (`test/Dialect/CSL/`) |
| `ninja check-csl-runtime` | CSL Runtime dialect (`test/Dialect/CSLRuntime/`) |
| `ninja check-csl-to-runtime` | CSL → CSL Runtime conversion (`test/Conversion/CSLToCSLRuntime/`) |
| `ninja check-csl-all` | All of the above |

Emit tests (e.g. `--emit-csl-rt`) live under `Dialect/CSL` (e.g. `emit.mlir`) and are run with `check-air-csl` or `check-csl-all`.

---

## Commands Reference

### Parse & Verify

```bash
# Parse CSL dialect
air-opt mlir/test/Dialect/CSL/kernel_ops.mlir

# Verify roundtrip (parse → print → parse)
air-opt mlir/test/Dialect/CSL/roundtrip.mlir --verify-roundtrip

# Verify runtime ops
air-opt mlir/test/Dialect/CSL/csl_runtime.mlir
```

### Convert

```bash
# Convert CSL to CSL Runtime
air-opt input.mlir -csl-to-csl-rt

# Save converted output
air-opt input.mlir -csl-to-csl-rt -o converted.mlir

# Apply conversion and show only csl_rt ops
air-opt input.mlir -csl-to-csl-rt | grep "csl_rt"
```

### Translate

```bash
# Emit CSL Runtime to Python (layout + run integrated in run.py)
air-opt input.mlir -csl-to-csl-rt | air-translate --emit-csl-rt -o output.py

# The emitted run.py integrates:
#   - platform = get_platform(args.cmaddr, config, target)
#   - layout = SdkLayout(platform)
#   - compile_artifacts = layout.compile(out_prefix='out')
#   - runtime = SdkRuntime(compile_artifacts, platform, memcpy_required=False)
# Extract the run.py section from output and run with: python run.py [--cmaddr ...] [--arch wse2|wse3]

# Check available translators
air-translate --help | grep emit-csl
```

---

## Debugging

### Check if --emit-csl still works (it shouldn't)

```bash
air-translate --emit-csl input.mlir
# Error: Unknown command line argument '--emit-csl'
```

This confirms the direct CSL→Text path has been removed.

### Verify dialect registration

```bash
# CSLRuntimeToPy.cpp should register both CSL and CSL Runtime
grep -A5 "DialectRegistry" mlir/lib/Targets/CSLRuntimeToPy.cpp
```

Expected: Both `xilinx::csl::CSLDialect` and `CSLRuntimeDialect` registered.

### Inspect conversion output

```bash
# See the actual CSL Runtime IR produced
air-opt input.mlir -csl-to-csl-rt | mlir-opt -print-op-stats
```

---

## Common Issues

### Issue: "Dialect 'csl' not found for custom op 'csl.kernel'"

**Cause**: The translator doesn't have CSL dialect registered.

**Solution**: Ensure `CSLRuntimeToPy.cpp` registers both dialects:
```cpp
registry.insert<xilinx::csl::CSLDialect, CSLRuntimeDialect, ...>();
```

### Issue: Conversion produces no CSL Runtime ops

**Cause**: The conversion pattern may not be matching.

**Solution**: Check that `CSLToCSLRuntime.cpp` has patterns for all layout ops:
- `csl.spatial_placement` → `csl_rt.create_layout` + ops
- `csl.code_region` → handled by conversion
- `csl.place` → `csl_rt.place`

### Issue: --emit-csl still available

**Cause**: The registration wasn't removed from `air-translate.cpp`.

**Solution**: Ensure `air-translate.cpp` doesn't call `registerCSLToTextTranslation()`.

---

## Test Files Location

| Test | File |
|------|------|
| CSL Parsing | `mlir/test/Dialect/CSL/kernel_ops.mlir` |
| CSL Roundtrip | `mlir/test/Dialect/CSL/roundtrip.mlir` |
| CSL Runtime | `mlir/test/Dialect/CSL/csl_runtime.mlir` |
| Conversion | `mlir/test/Conversion/CSLToCSLRuntime/layout_to_runtime.mlir` |
| Full Pipeline | `mlir/test/Dialect/CSL/emit.mlir` |

---

## Implementation Details

### Files Modified

| File | Change |
|------|--------|
| `air-translate/air-translate.cpp` | Removed `registerCSLToTextTranslation()` |
| `include/air/Dialect/CSL/CSLDialect.h` | Deprecated direct registration |
| `lib/Targets/CSLRuntimeToPy.cpp` | Added CSL dialect registration |
| `lib/Conversion/CSLToCSLRuntime/CSLToCSLRuntime.cpp` | Handles conversion |
| `test/Dialect/CSL/emit.mlir` | Updated to use conversion pipeline |

### Key Design Decisions

1. **No direct CSL→Text**: Forces discipline, ensures conversion is applied
2. **Mandatory conversion**: All spatial ops must go through `-csl-to-csl-rt`
3. **Single emitter**: `--emit-csl-rt` is the only output path
4. **PE ops preserved**: CSL kernels remain in IR through conversion

---

## Related Documentation

- [CSL Dialect Reference](CSL_Dialect_Reference.md)
- [CSL Runtime Specification](CSL_RUNTIME_DIALECT_SPEC.md)
- [AIR to CSL Design](design_air_to_csl.md)
- [CLAUDE.md](../../CLAUDE.md) - Build and environment setup

