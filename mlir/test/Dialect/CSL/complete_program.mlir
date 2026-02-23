//===- complete_program.mlir - CSL complete program test -------*- MLIR -*-===//
//
// Part of the air-to-csl project.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
//
// Integration test: a complete CSL program structure representing a simple
// GEMV kernel mapped to a 2x2 PE grid. Validates that all CSL sub-dialect
// categories (layout, placement, routing, kernel, data movement, runtime)
// compose correctly in a single module.
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s | FileCheck %s

// CHECK: csl.layout {
// CHECK:   csl.set_rectangle 2, 2
// CHECK:   csl.set_tile_code 0, 0 file("pe_program.csl")
// CHECK:   csl.set_tile_code 1, 0 file("pe_program.csl")
// CHECK:   csl.set_tile_code 0, 1 file("pe_program.csl")
// CHECK:   csl.set_tile_code 1, 1 file("pe_program.csl")
// CHECK:   csl.export_name "arg_0" : memref<24xf32>
// CHECK:   csl.export_name "arg_1" : memref<6xf32>
// CHECK:   csl.export_name "arg_2" : memref<4xf32>
// CHECK:   csl.export_name "init_and_compute" : () -> ()
// CHECK: }
csl.layout {
  csl.set_rectangle 2, 2
  csl.set_tile_code 0, 0 file("pe_program.csl")
  csl.set_tile_code 1, 0 file("pe_program.csl")
  csl.set_tile_code 0, 1 file("pe_program.csl")
  csl.set_tile_code 1, 1 file("pe_program.csl")
  csl.export_name "arg_0" : memref<24xf32>
  csl.export_name "arg_1" : memref<6xf32>
  csl.export_name "arg_2" : memref<4xf32>
  csl.export_name "init_and_compute" : () -> ()
}

// CHECK: csl.module @pe_program {
// CHECK:   csl.param @memcpy_params : i64
// CHECK:   csl.var @arg_0 : memref<24xf32>
// CHECK:   csl.var @arg_1 : memref<6xf32>
// CHECK:   csl.var @arg_2 : memref<4xf32>
// CHECK:   csl.func @compute()
// CHECK:   csl.func @init_and_compute()
// CHECK:   csl.comptime {
// CHECK:     csl.export_symbol @arg_0 alias("arg_0")
// CHECK:     csl.export_symbol @arg_1 alias("arg_1")
// CHECK:     csl.export_symbol @arg_2 alias("arg_2")
// CHECK:     csl.export_symbol @init_and_compute
// CHECK:   }
// CHECK: }
csl.module @pe_program {
  csl.param @memcpy_params : i64
  csl.var @arg_0 : memref<24xf32>
  csl.var @arg_1 : memref<6xf32>
  csl.var @arg_2 : memref<4xf32>

  csl.func @compute() {
    csl.return
  }

  csl.func @init_and_compute() {
    csl.return
  }

  csl.comptime {
    csl.export_symbol @arg_0 alias("arg_0")
    csl.export_symbol @arg_1 alias("arg_1")
    csl.export_symbol @arg_2 alias("arg_2")
    csl.export_symbol @init_and_compute
  }
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