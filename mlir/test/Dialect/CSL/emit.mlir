//===- emit.mlir - CSL Runtime translation tests -----------*- MLIR -*-===//
//
// Part of the air-to-csl project.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
//
// Tests for air-translate --emit-csl-rt: CSL → CSL Runtime → Python files.
//
// Pipeline: CSL dialect with spatial_placement
//           → (-csl-to-csl-rt pass)
//           → CSL Runtime dialect
//           → (--emit-csl-rt)
//           → layout.py + run.py + kernel.csl
//
//===----------------------------------------------------------------------===//

// ============================================================================
// Test: CSL spatial_placement round-trip
//   Verifies: kernel, color, route, code_region, place parsing
//
// RUN: air-opt %s | FileCheck %s
// ============================================================================

// CHECK: func.func @main
func.func @main() {
  %k = csl.kernel {
  } {source_file = "pe_program.csl"} : !csl.kernel

  csl.spatial_placement {
    // CHECK: csl.color
    %c = csl.color : !csl.color
    // CHECK: csl.route
    %r = csl.route in(RAMP) out(EAST) : !csl.route
    // CHECK: csl.code_region routes
    %region = csl.code_region routes(%r) colors(%c) {
    } {width = 2 : i64, height = 2 : i64} : !csl.code_region

    // CHECK: csl.place
    csl.place %region %k {x = 0 : i64, y = 0 : i64}
  }
  return
}
