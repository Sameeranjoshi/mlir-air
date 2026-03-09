//===- placement_ops.mlir - CSL placement ops tests -----------*- MLIR -*-===//
//
// Part of the air-to-csl project.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
//
// Tests for csl.place ops. 
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s | FileCheck %s

// CHECK-LABEL: module {

// Basic placement with kernel binding
// CHECK: csl.spatial_placement {
// CHECK:   %[[R:.*]] = csl.route in(RAMP) out(EAST) : !csl.route
// CHECK:   %[[C:.*]] = csl.color : !csl.color
// CHECK:   %[[REG:.*]] = csl.code_region
// CHECK:   %[[K:.*]] = csl.kernel
// CHECK:   csl.place %[[REG]] %[[K]] {x = 0 : i64, y = 0 : i64}
// CHECK: }
csl.spatial_placement {
  %r  = csl.route in(RAMP) out(EAST) : !csl.route
  %c  = csl.color : !csl.color
  %reg = csl.code_region routes(%r) colors(%c) {
  } {width = 2 : i64, height = 2 : i64} : !csl.code_region
  %k = csl.kernel {
  } {source_file = "pe_program.csl"} : !csl.kernel
  csl.place %reg %k {x = 0 : i64, y = 0 : i64}
}

// Placement with kernel params
// CHECK: csl.spatial_placement {
// CHECK:   %[[K:.*]] = csl.kernel
// CHECK: }
csl.spatial_placement {
  %r  = csl.route in(RAMP) out(EAST) : !csl.route
  %c  = csl.color : !csl.color
  %reg = csl.code_region routes(%r) colors(%c) {
  } {width = 2 : i64, height = 2 : i64} : !csl.code_region
  %k = csl.kernel {
  } {source_file = "pe_program.csl"} : !csl.kernel
  csl.place %reg %k {x = 0 : i64, y = 0 : i64}
}
