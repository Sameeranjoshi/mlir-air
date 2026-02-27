//===- roundtrip.mlir - CSL dialect round-trip tests -----------*- MLIR -*-===//
//
// Part of the air-to-csl project.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
//
// Verifies that the CSL dialect ops survive a parse-print-parse round trip
// using --verify-roundtrip.
//
//===----------------------------------------------------------------------===//

// RUN: air-opt --verify-roundtrip %s | FileCheck %s

// ---- Spatial placement with routing and kernels ----

// CHECK-LABEL: csl.spatial_placement
// CHECK:   %[[C:.*]] = csl.color : !csl.color
// CHECK:   %[[R:.*]] = csl.route in(RAMP) out(EAST) : i32
// CHECK:   %[[K:.*]] = csl.kernel "sender.csl"
// CHECK:   %[[REG:.*]] = csl.code_region routes(%[[R]]) colors(%[[C]]) shape(4, 4)
// CHECK:   } : !csl.code_region
// CHECK:   csl.place %[[REG]] at(0, 0) kernel(%[[K]])
csl.spatial_placement {
  %c = csl.color : !csl.color
  %r = csl.route in(RAMP) out(EAST) : i32
  %k = csl.kernel "sender.csl" params({color_id = 0 : i32}) {
    csl.var @data : memref<128xf32>
    csl.func @send() { csl.return }
  } : !csl.kernel
  %reg = csl.code_region routes(%r) colors(%c) shape(4, 4) {
    csl.paint pe(0, 0) route(%r) color(%c)
  } : !csl.code_region
  csl.place %reg at(0, 0) kernel(%k)
}

// ---- Module with tasks (standalone, not in spatial_placement) ----

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

// ---- Color with explicit ID ----

// CHECK: %[[C0:.*]] = csl.color 0 : !csl.color
%c0 = csl.color 0 : !csl.color

// ---- Route directions ----

// CHECK: %[[R1:.*]] = csl.route in(RAMP) out(EAST) : i32
// CHECK: %[[R2:.*]] = csl.route in(WEST) out(RAMP) : i32
%r1 = csl.route in(RAMP) out(EAST) : i32
%r2 = csl.route in(WEST) out(RAMP) : i32

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
