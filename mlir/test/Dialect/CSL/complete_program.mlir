//===- complete_program.mlir - CSL complete program test -------*- MLIR -*-===//
//
// Part of the air-to-csl project.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
//
// Integration test: a complete CSL program structure using the spatial
// placement model with kernels, routes, colors, and placement.
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s | FileCheck %s

// CHECK-LABEL: csl.spatial_placement
// CHECK:   %[[RED:.*]] = csl.color : !csl.color
// CHECK:   %[[R1:.*]] = csl.route in(RAMP) out(EAST) : !csl.route
// CHECK:   %[[K:.*]] = csl.kernel
// CHECK:   %[[REG:.*]] = csl.code_region routes(%[[R1]]) colors(%[[RED]])
// CHECK:   {width = 2 : i64, height = 2 : i64} : !csl.code_region
// CHECK:   csl.place %[[REG]] %[[K]] {x = 0 : i64, y = 0 : i64}
csl.spatial_placement {
  %red = csl.color : !csl.color
  %r1 = csl.route in(RAMP) out(EAST) : !csl.route
  %k = csl.kernel {
  } {source_file = "pe_program.csl"} : !csl.kernel
  %region = csl.code_region routes(%r1) colors(%red) {
  } {width = 2 : i64, height = 2 : i64} : !csl.code_region
  csl.place %region %k {x = 0 : i64, y = 0 : i64}
}

// A single PE has 3 main components:(CSL code)
// 0.TILE SELECTION:DATA(Memory)|COMPUTE|COMMUNICATION tile
  // Say make this tile a memory tile only
  // Say make this a compute only on external data, no internal data
// 1.DATA
// 2.COMPUTE
  // functions
  // tasks
// 3.SCHEDULE/ALGORITHM
  // Wait until task1 is finished
  // Now perform task2
  // This can be a state machine maybe?
  // in a loop:
    // Do task1


// Layout block:
  // Layout and placement (python SDKLayout)

// Routing and colors (python SDKRouting)

// Runtime Host (python SDKRuntime)
