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

      // Task triggered by a local task id
      // CHECK: csl.task @exit_task attributes {id = 8 : i32, trigger_kind = "local_task_id"} {
      // CHECK:   csl.return
      // CHECK: }
      csl.task @exit_task attributes {trigger_kind = "local_task_id", id = 8 : i32} {
        csl.return
      }

      // Task bound to a color (symbol-referenced)
      // CHECK: csl.task @recv_task attributes {color = @recv_color, trigger_kind = "color"} {
      // CHECK:   csl.return
      // CHECK: }
      csl.task @recv_task attributes {trigger_kind = "color", color = @recv_color} {
        csl.return
      }

      // Task triggered by a higher local task id
      // CHECK: csl.task @data_task attributes {id = 15 : i32, trigger_kind = "local_task_id"} {
      // CHECK:   csl.return
      // CHECK: }
      csl.task @data_task attributes {trigger_kind = "local_task_id", id = 15 : i32} {
        csl.return
      }
    }
    csl.layout {width = 1 : i64, height = 1 : i64} @test_layout {
      csl.color @recv_color
      csl_layout.place @test_pe at (0, 0)
    }
  }
}
