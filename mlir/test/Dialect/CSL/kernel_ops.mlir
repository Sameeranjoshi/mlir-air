//===- kernel_ops.mlir - CSL kernel dialect ops tests ----------*- MLIR -*-===//
//
// Part of the air-to-csl project.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s | FileCheck %s

// CHECK-LABEL: csl.wafer @test_wafer
// CHECK: csl.program @test_pe

module {
  csl.wafer @test_wafer {arch = "wse3"} {
    csl.program @test_pe {

      // Empty function
      // CHECK: csl.func @noop {
      // CHECK:   csl.return
      // CHECK: }
      csl.func @noop {
        csl.return
      }

      // Function with compute body
      // CHECK: csl.func @compute {
      // CHECK:   csl.return
      // CHECK: }
      csl.func @compute {
        csl.return
      }

      // Multiple functions
      // CHECK: csl.func @init {
      // CHECK:   csl.return
      // CHECK: }
      // CHECK: csl.func @finalize {
      // CHECK:   csl.return
      // CHECK: }
      csl.func @init {
        csl.return
      }
      csl.func @finalize {
        csl.return
      }

      // NOTE: csl.task tests previously here used csl.color SSA values from
      // within csl.program. After Task 1 (color scope move), csl.color lives
      // in csl.layout and is referenced by symbol. Task 4 will refactor
      // csl.task to take a FlatSymbolRefAttr and reinstate these tests.
    }
  }
}
