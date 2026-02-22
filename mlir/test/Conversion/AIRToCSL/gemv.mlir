//===- gemv.mlir -  AIR to CSL GEMV lowering test -------------*- MLIR -*-===//
//
// Copyright (C) 2022, Xilinx Inc. All rights reserved.
// Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-to-csl="output-dir=%t" && cat %t/layout.csl | FileCheck %s --check-prefix=LAYOUT
// RUN: cat %t/pe_program.csl | FileCheck %s --check-prefix=PE

// A GEMV-like AIR program with a 2x2 herd.

// LAYOUT: @set_rectangle(2, 2)
// LAYOUT: @set_tile_code(0, 0, "pe_program.csl"
// LAYOUT: @set_tile_code(1, 0, "pe_program.csl"
// LAYOUT: @set_tile_code(0, 1, "pe_program.csl"
// LAYOUT: @set_tile_code(1, 1, "pe_program.csl"

// PE: param memcpy_params: comptime_struct
// PE: fn compute()
// PE: fn init_and_compute()

module {
  func.func @gemv(%A: memref<24xf32>, %x: memref<6xf32>, %y: memref<4xf32>) {
    %c1 = arith.constant 1 : index
    air.launch (%tx) in (%sx=%c1) args(%a0=%A, %a1=%x, %a2=%y) : memref<24xf32>, memref<6xf32>, memref<4xf32> {
      air.segment @seg0 args(%s0=%a0, %s1=%a1, %s2=%a2) : memref<24xf32>, memref<6xf32>, memref<4xf32> {
        %c2 = arith.constant 2 : index
        air.herd @herd0 tile(%htx, %hty) in (%hsx=%c2, %hsy=%c2) args(%h0=%s0, %h1=%s1, %h2=%s2) : memref<24xf32>, memref<6xf32>, memref<4xf32> {
          %zero = arith.constant 0 : index
          %v = memref.load %h1[%zero] : memref<6xf32>
          %w = memref.load %h0[%zero] : memref<24xf32>
          %prod = arith.mulf %v, %w : f32
          memref.store %prod, %h2[%zero] : memref<4xf32>
          air.herd_terminator
        }
        air.segment_terminator
      }
      air.launch_terminator
    }
    return
  }
}
