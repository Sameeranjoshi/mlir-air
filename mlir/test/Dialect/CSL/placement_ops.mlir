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
// CHECK:   %[[R:.*]] = csl.route in(RAMP) out(EAST) : i32
// CHECK:   %[[C:.*]] = csl.color : !csl.color
// CHECK:   %[[REG:.*]] = csl.code_region
// CHECK:   %[[K:.*]] = csl.kernel "pe_program.csl"
// CHECK:   csl.place %[[REG]] at(0, 0) kernel(%[[K]])
// CHECK: }
csl.spatial_placement {
  %r  = csl.route in(RAMP) out(EAST) : i32
  %c  = csl.color : !csl.color
  %reg = csl.code_region routes(%r) colors(%c) shape(2, 2) {
  } : !csl.code_region
  %k = csl.kernel "pe_program.csl" {
  } : !csl.kernel
  csl.place %reg at(0, 0) kernel(%k)
}

// Placement with kernel params
// CHECK: csl.spatial_placement {
// CHECK:   %[[K:.*]] = csl.kernel "pe_program.csl" params({col = 0 : i32, row = 0 : i32})
// CHECK: }
csl.spatial_placement {
  %r  = csl.route in(RAMP) out(EAST) : i32
  %c  = csl.color : !csl.color
  %reg = csl.code_region routes(%r) colors(%c) shape(1, 1) {
  } : !csl.code_region
  %k = csl.kernel "pe_program.csl" params({col = 0 : i32, row = 0 : i32}) {
  } : !csl.kernel
  csl.place %reg at(0, 0) kernel(%k)
}
