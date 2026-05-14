# CSL + Shardy Mixed-Dialect Integration Plan

**Date**: 2026-03-09
**Context**: Mixed CSL + Shardy IR for declarative data distribution with Cerebras uniform constraint
**Status**: Implementation ready (phase-by-phase approach)
**Approach**: No new dialect; reuse Shardy annotations within CSL modules

---

## Vision: Mixed CSL + Shardy IR

**Goal**: Write CSL programs that declare data with Shardy sharding specs, kernels with CSL placement, all in the same module.

```mlir
module {
  // Data declarations with uniform distribution specs (Shardy)
  %matrix = csl.var : tensor<100x100xf32>
  %sharding = shard.sharding(%matrix) {split_axes = [0: 2, 1: 3]}

  // Grid definition (Shardy)
  %grid = shard.grid(2, 3)  // 2×3 = 6 tiles

  // Kernel definition (CSL)
  %kernel = csl.kernel "pe.csl" {
    csl.func @compute() : () -> () {
      // Use %matrix here; sharding constrains its layout
      ...
    }
  }

  // Spatial placement (CSL respects Shardy sharding)
  csl.place %kernel at (%x, %y) on_grid %grid

  // Data movement ops respect sharding (CSL)
  %tile_slice = csl.dma_memcpy %matrix[#affine_map<(i,j)->(i,j)>]
}
```

---

## Design Principles

### 1. **Shardy Handles "What" (Data Distribution)**
- `shard.sharding` declares how tensors map to grid dimensions
- `shard.grid` defines the device mesh (2D for Cerebras WSE)
- Guarantees: uniform partitioning, compile-time verification

### 2. **CSL Handles "Where" (Spatial Placement)**
- `csl.place` maps kernels to grid coordinates
- `csl.var` declares variables (type, shape, name)
- Kernels consume sharded data according to their tile location

### 3. **Automatic Slice Derivation**
- Grid coordinates + sharding spec = which tensor slice each tile gets
- **Example**: Tile (x, y) on a 2×3 grid with `split_axes = [0:2, 1:3]`
  - Tensor slice: `matrix[x*(100/2):(x+1)*(100/2), y*(100/3):(y+1)*(100/3)]`
  - Automatically computed from coordinates + sharding

### 4. **Validation Layer**
- Verify CSL placements fit within Shardy grid dimensions
- Verify tensor dims are divisible by split_axes counts
- Verify no memory conflicts (two tiles claim same data)

---

## IR Structure: Concrete Examples

### Example 1: Simple 2D Matmul with Uniform Distribution

```mlir
module {
  // Data with sharding
  %a = csl.var : tensor<100x100xf32> {"a"}
  %b = csl.var : tensor<100x100xf32> {"b"}
  %c = csl.var : tensor<100x100xf32> {"c"}

  %shard_a = shard.sharding(%a) {split_axes = [0: 2, 1: 2]}  // 4 tiles (2×2)
  %shard_b = shard.sharding(%b) {split_axes = [0: 2, 1: 2]}
  %shard_c = shard.sharding(%c) {split_axes = [0: 2, 1: 2]}

  // Grid matches sharding: 2×2 tiles
  %grid = shard.grid(2, 2)

  // Kernel (same for all tiles)
  %kernel = csl.kernel "pe.csl" {
    csl.func @matmul() : () -> () {
      // Kernel receives tile coordinates as implicit context
      // Can access: a_tile, b_tile, c_tile (slices from sharding)
      ...
    }
  }

  // Place kernel on all grid positions
  csl.place %kernel at (0, 0) on_grid %grid
  csl.place %kernel at (0, 1) on_grid %grid
  csl.place %kernel at (1, 0) on_grid %grid
  csl.place %kernel at (1, 1) on_grid %grid
}
```

**Derived data layout**:
- Tile (0, 0) gets: `a[0:50, 0:50]`, `b[0:50, 0:50]`, `c[0:50, 0:50]`
- Tile (0, 1) gets: `a[0:50, 50:100]`, `b[0:50, 50:100]`, `c[0:50, 50:100]`
- (etc. for tiles 1,0 and 1,1)

All tiles compute **identical code** on different data slices. ✅ Uniform distribution.

### Example 2: Broadcast (Replicate Data)

```mlir
module {
  %weights = csl.var : tensor<64xf32> {"weights"}

  // No split axes = replicate on all tiles
  %shard_weights = shard.sharding(%weights) {split_axes = []}

  %grid = shard.grid(4, 4)  // 16 tiles

  %kernel = csl.kernel "pe.csl" {
    csl.func @inference() : () -> () {
      // All tiles get full weights[0:64]
      ...
    }
  }

  // Place on all tiles
  csl.place %kernel at (...) on_grid %grid  // fill all coordinates
}
```

---

## Implementation Phases

### Phase 0: Enable Shardy Dialec in Build (DONE)
- Shardy is already part of LLVM; just include `#include "mlir/Dialect/Shard/ShardDialect.h"`
- Register in CMakeLists.txt

### Phase 1: Extend CSL IR (Minimal)

**Changes to CSLOps.td:**
- Add operand to `csl.place`: optional `on_grid` grid operand
  ```
  let arguments = (ins SymbolRefAttr:$kernel, Index:$x, Index:$y,
                       Optional<AnyType>:$grid);  // new
  ```
- Keep `csl.var` unchanged (already holds tensor types)
- No changes needed to `csl.route`, `csl.dma_memcpy`

**Changes to CSLBase.td:**
- Add traits/interfaces for grid compatibility checking

**New in CSLDialect.h/cpp:**
- Helper: `derived_slice_t computeTileSlice(grid_dims, split_axes, tile_coords)`
- Verification: `verifyCSLPlacementConsistency(module)` pass

### Phase 2: Validation Pass (~300 lines)

**New file**: `mlir/lib/Conversion/CSLToCSLRuntime/CSLShartyValidator.cpp`

```cpp
// Pseudo-code
class CSLShartyValidator : public OperationPass<ModuleOp> {
  void runOnOperation() override {
    // 1. Collect all shard.sharding ops and their constraints
    auto shardings = module.getOps<shard::ShardingOp>();

    // 2. Verify divisibility: tensor_dim % split_count == 0
    for (auto shard : shardings) {
      verifyDivisibility(shard);
    }

    // 3. Collect csl.place ops and verify against grid
    auto placements = module.getOps<csl::PlaceOp>();
    for (auto place : placements) {
      verifyPlacementInGrid(place, grid_dims);
    }

    // 4. Verify no tile claims multiple data slices (memory conflict)
    verifyMemoryNonOverlap(shardings, placements);
  }
};
```

### Phase 3: CSLToCSLRuntime Enhancement (~200 lines)

**Update `mlir/lib/Conversion/CSLToCSLRuntime/CSLToCSLRuntime.cpp`:**

```cpp
// When lowering csl.var with shard.sharding, generate Python:
// Before:  SdkLayout.addVariable(IntVector([100, 100]), "matrix")
// After:   for (x, y) in grid.iter():
//            SdkLayout.addVariable(IntVector([50, 50]), "matrix",
//                                  placement=Placement(x, y))

void lowerCslVarWithSharding(csl::VarOp var, shard::ShardingOp sharding) {
  auto slice_dims = computeTileDims(var.type(), sharding.split_axes());

  // Generate per-tile placement code
  for (auto [tile_coords, _] : grid.iter()) {
    emitPy(llvm::format(
      "layout.addVariable(IntVector([%s]), \"%s\", placement=Placement(%s))",
      slice_dims, var.name(), tile_coords
    ));
  }
}
```

### Phase 4: Test Suite (~500 lines)

**New test file**: `mlir/test/Integration/CSLShardy/matmul_2x2.mlir`

```mlir
// RUN: air-opt %s --csl-shardy-validate --convert-csl-to-csl-runtime | \
//       air-translate --emit-csl-rt --csl-output-dir=/tmp/out

module {
  %a = csl.var : tensor<100x100xf32> {"a"}
  %shard_a = shard.sharding(%a) {split_axes = [0: 2, 1: 2]}
  %grid = shard.grid(2, 2)

  %kernel = csl.kernel "pe.csl" { ... }
  csl.place %kernel at (0, 0) on_grid %grid
  csl.place %kernel at (0, 1) on_grid %grid
  csl.place %kernel at (1, 0) on_grid %grid
  csl.place %kernel at (1, 1) on_grid %grid
}
```

Expected output: `layout.py` with 4 addVariable calls, each with proper placement.

---

## Error Handling & Edge Cases

### Case 1: Non-Divisible Tensor
```mlir
%tensor : tensor<100x100xf32>
%shard = shard.sharding(%tensor) {split_axes = [0: 3]}  // 100 % 3 != 0
```
**Error**: "Tensor dimension 100 not divisible by split axis count 3"

**Solution**: User must pad or tile manually
```mlir
%padded = tensor.pad %tensor low [0] high [2] {fill_value: 0.0}
         // ⟹ 102x100, now divisible by 3
%shard = shard.sharding(%padded) {split_axes = [0: 3]}
```

### Case 2: Placement Out of Bounds
```mlir
%grid = shard.grid(2, 2)  // 0-1 on each axis
%kernel = ...
csl.place %kernel at (3, 4) on_grid %grid  // INVALID
```
**Error**: "Placement coordinate (3, 4) exceeds grid bounds (2, 2)"

### Case 3: Missing Sharding Spec
```mlir
%var = csl.var : tensor<100xf32>
// no shard.sharding(...)
csl.place %kernel at (0) on_grid %grid
```
**Warning** (not error): "Variable %var has no sharding spec; assuming replication"

---

## Code Locations & Artifacts

| Component | File(s) | Lines | Status |
|-----------|---------|-------|--------|
| CSL IR extension | mlir/include/air/Dialect/CSL/CSLOps.td | ~20 | Phase 1 |
| Validation pass | mlir/lib/Conversion/CSLToCSLRuntime/CSLShartyValidator.cpp | ~300 | Phase 2 |
| Runtime lowering | mlir/lib/Conversion/CSLToCSLRuntime/CSLToCSLRuntime.cpp | ~200 (added) | Phase 3 |
| Test suite | mlir/test/Integration/CSLShardy/ | ~500 | Phase 4 |
| CMakeLists.txt | mlir/lib/Conversion/CSLToCSLRuntime/ | ~5 (register pass) | Phase 1 |

**Total implementation**: ~1025 lines, phased into 4 increments.

---

## Benefits Achieved

✅ **Uniform Distribution Guarantee**: Shardy's `split_axes` enforces mathematical uniformity
✅ **Compile-Time Verification**: Catch divisibility/bounds errors before execution
✅ **Declarative Data Layout**: No imperative loop-based mapping (cf. Graphcore)
✅ **Minimal CSL Changes**: Only add optional grid operand to `csl.place`
✅ **Reuse Shardy Ecosystem**: Benefit from GSPMD-inspired validation + future optimizations
✅ **Mixed-Dialect IR**: No new dialect needed; CSL + Shardy coexist naturally

---

## Rollout Strategy

1. **Phase 1 (CSL IR)**: Backward compatible; old CSL code still works (grid operand optional)
2. **Phase 2 (Validator)**: Optional pass; doesn't affect existing pipelines
3. **Phase 3 (Runtime)**: Auto-enabled if sharding specs detected
4. **Phase 4 (Tests)**: Showcase best practices; encourage adoption

---

## Success Criteria

- ✅ CSL + Shardy can coexist in same module
- ✅ Validator catches all divisibility/bounds violations
- ✅ CSL→CSLRuntime generates correct per-tile data layouts
- ✅ Integration tests pass (2×2, 4×4, broadcast patterns)
- ✅ Existing CSL tests unaffected

---

## References

- [MLIR Shardy Dialect](https://mlir.llvm.org/docs/Dialects/Shard/)
- [CSL ODS Current](mlir/include/air/Dialect/CSL/CSLOps.td)
- [CSL Runtime Lowering](mlir/lib/Conversion/CSLToCSLRuntime/CSLToCSLRuntime.cpp)
