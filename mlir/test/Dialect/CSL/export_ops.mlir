//===- export_ops.mlir - CSL export op round-trip tests --------*- MLIR -*-===//
//
// Part of the air-to-csl project.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
//
// Round-trip tests for csl.export_name and csl.export_symbol.
// These ops are the Phase 1 additions for the AIR → CSL vecadd milestone.
//
// Note on assembly format: the declarative format for csl.export_name prints
// the optional direction group as {direction = "..."} with no leading space,
// immediately after the type attribute.
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s | air-opt | FileCheck %s

// ============================================================================
// csl.export_name — buffer with direction "in"
// ============================================================================

// CHECK-LABEL: func.func @export_name_in
func.func @export_name_in() {
  // CHECK: csl.export_name "a" : memref<256xf32>{direction = "in"}
  csl.export_name "a" : memref<256xf32>{direction = "in"}
  return
}

// ============================================================================
// csl.export_name — buffer with direction "out"
// ============================================================================

// CHECK-LABEL: func.func @export_name_out
func.func @export_name_out() {
  // CHECK: csl.export_name "c" : memref<256xf32>{direction = "out"}
  csl.export_name "c" : memref<256xf32>{direction = "out"}
  return
}

// ============================================================================
// csl.export_name — function type (no direction)
// ============================================================================

// CHECK-LABEL: func.func @export_name_fn
func.func @export_name_fn() {
  // CHECK: csl.export_name "compute" : () -> ()
  csl.export_name "compute" : () -> ()
  return
}

// ============================================================================
// csl.export_symbol — with alias
// ============================================================================

// CHECK-LABEL: func.func @export_symbol_with_alias
func.func @export_symbol_with_alias() {
  %k = csl.kernel {
    csl.var @x : memref<8xf32>
    // CHECK: csl.export_symbol @x alias("a")
    csl.export_symbol @x alias("a")
  } {source_file = "k.csl"} : !csl.kernel
  return
}

// ============================================================================
// csl.export_symbol — without alias
// ============================================================================

// CHECK-LABEL: func.func @export_symbol_no_alias
func.func @export_symbol_no_alias() {
  %k = csl.kernel {
    csl.func @compute {
      csl.return
    }
    // CHECK: csl.export_symbol @compute
    csl.export_symbol @compute
  } {source_file = "k.csl"} : !csl.kernel
  return
}

// ============================================================================
// Combined: multiple export_name ops
// ============================================================================

// CHECK-LABEL: func.func @export_name_combined
func.func @export_name_combined() {
  // CHECK: csl.export_name "a" : memref<256xf32>{direction = "in"}
  csl.export_name "a" : memref<256xf32>{direction = "in"}
  // CHECK: csl.export_name "c" : memref<256xf32>{direction = "out"}
  csl.export_name "c" : memref<256xf32>{direction = "out"}
  // CHECK: csl.export_name "compute" : () -> ()
  csl.export_name "compute" : () -> ()
  return
}
