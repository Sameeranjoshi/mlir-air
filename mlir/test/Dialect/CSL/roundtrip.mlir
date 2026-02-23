//===- roundtrip.mlir - CSL dialect round-trip tests -----------*- MLIR -*-===//
//
// Part of the air-to-csl project.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
//
// Verifies that the CSL dialect ops survive a parse-print-parse round trip
// using --verify-roundtrip. This ensures the custom parsers and printers
// are fully consistent.
//
//===----------------------------------------------------------------------===//

// RUN: air-opt --verify-roundtrip %s | FileCheck %s

// ---- Layout + Placement ----

// CHECK: csl.layout {
// CHECK:   csl.set_rectangle 4, 4
// CHECK:   csl.set_tile_code 0, 0 file("sender.csl") params({color_id = 0 : i32})
// CHECK:   csl.set_tile_code 1, 0 file("receiver.csl") params({color_id = 0 : i32})
// CHECK:   csl.export_name "input" : memref<128xf32>
// CHECK:   csl.export_name "output" : memref<128xf32>
// CHECK:   csl.export_name "run" : () -> ()
// CHECK: }
csl.layout {
  csl.set_rectangle 4, 4
  csl.set_tile_code 0, 0 file("sender.csl") params({color_id = 0 : i32})
  csl.set_tile_code 1, 0 file("receiver.csl") params({color_id = 0 : i32})
  csl.export_name "input" : memref<128xf32>
  csl.export_name "output" : memref<128xf32>
  csl.export_name "run" : () -> ()
}

// ---- Module with tasks, routing, data movement ----

// CHECK: csl.module @sender {
// CHECK:   csl.param @color_id : i32
// CHECK:   csl.var @data : memref<128xf32>
// CHECK:   csl.func @send()
// CHECK:   csl.task @send_complete() color(1) {
// CHECK:     csl.return
// CHECK:   }
// CHECK:   csl.comptime {
// CHECK:     csl.export_symbol @data alias("input")
// CHECK:     csl.export_symbol @send alias("run")
// CHECK:   }
// CHECK: }
csl.module @sender {
  csl.param @color_id : i32
  csl.var @data : memref<128xf32>

  csl.func @send() {
    csl.return
  }

  csl.task @send_complete() color(1) {
    csl.return
  }

  csl.comptime {
    csl.export_symbol @data alias("input")
    csl.export_symbol @send alias("run")
  }
}

// CHECK: csl.module @receiver {
// CHECK:   csl.param @color_id : i32
// CHECK:   csl.var @result : memref<128xf32>
// CHECK:   csl.task @recv() color(0) {
// CHECK:     csl.return
// CHECK:   }
// CHECK:   csl.comptime {
// CHECK:     csl.export_symbol @result alias("output")
// CHECK:   }
// CHECK: }
csl.module @receiver {
  csl.param @color_id : i32
  csl.var @result : memref<128xf32>

  csl.task @recv() color(0) {
    csl.return
  }

  csl.comptime {
    csl.export_symbol @result alias("output")
  }
}

// ---- Routing configuration (top-level) ----

// CHECK: %[[C0:.*]] = csl.color 0
// CHECK: %[[C1:.*]] = csl.color 1
// CHECK: csl.route %[[C0]] dir(EAST)
// CHECK: csl.route %[[C1]] dir(WEST)
%send_color = csl.color 0 : !csl.color
%ack_color = csl.color 1 : !csl.color
csl.route %send_color dir(EAST)
csl.route %ack_color dir(WEST)

// ---- Data movement in a function ----

// CHECK-LABEL: func.func @test_dsd_pipeline
func.func @test_dsd_pipeline(%buf_in : memref<256xf32>, %buf_out : memref<256xf32>) {
  %len = arith.constant 256 : index
  %c = csl.color 2 : !csl.color

  // CHECK: csl.get_mem_dsd
  %src_dsd = csl.get_mem_dsd %buf_in, %len : memref<256xf32>, index -> !csl.dsd
  %dst_dsd = csl.get_mem_dsd %buf_out, %len : memref<256xf32>, index -> !csl.dsd

  // CHECK: csl.get_fab_dsd fabin
  %fab_in = csl.get_fab_dsd fabin %c, %len : !csl.color, index -> !csl.dsd
  // CHECK: csl.get_fab_dsd fabout
  %fab_out = csl.get_fab_dsd fabout %c, %len : !csl.color, index -> !csl.dsd

  // CHECK: csl.mov
  csl.mov %dst_dsd, %fab_in : !csl.dsd, !csl.dsd
  csl.mov %fab_out, %src_dsd : !csl.dsd, !csl.dsd

  return
}
