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
// CHECK:   %[[R1:.*]] = csl.route in(RAMP) out(EAST) : i32
// CHECK:   %[[K:.*]] = csl.kernel "pe_program.csl"
// CHECK:   %[[REG:.*]] = csl.code_region routes(%[[R1]]) colors(%[[RED]]) shape(2, 2)
// CHECK:     csl.paint pe(0, 0) route(%[[R1]]) color(%[[RED]])
// CHECK:   } : !csl.code_region
// CHECK:   csl.place %[[REG]] at(0, 0) kernel(%[[K]])
csl.spatial_placement {
  %red = csl.color : !csl.color
  %r1 = csl.route in(RAMP) out(EAST) : i32
  %k = csl.kernel "pe_program.csl" params({memcpy_params = 0 : i64}) {
    csl.var @arg_0 : memref<24xf32>
    csl.var @arg_1 : memref<6xf32>
    csl.var @arg_2 : memref<4xf32>
    csl.func @compute() : () -> () { csl.return }
    csl.func @init_and_compute() : () -> () { csl.return }
    csl.comptime {
      csl.export_symbol @arg_0 alias("arg_0")
      csl.export_symbol @arg_1 alias("arg_1")
      csl.export_symbol @arg_2 alias("arg_2")
      csl.export_symbol @init_and_compute
    }
  } : !csl.kernel
  %region = csl.code_region routes(%r1) colors(%red) shape(2, 2) {
    csl.paint pe(0, 0) route(%r1) color(%red)
  } : !csl.code_region
  csl.place %region at(0, 0) kernel(%k)
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
