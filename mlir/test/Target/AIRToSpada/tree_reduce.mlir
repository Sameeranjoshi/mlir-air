//===- tree_reduce.mlir ----------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
// Log-depth tree reduction on a line of 8 tiles: per-stage channels whose
// edges are uniform 2^s-hop jumps become parity streams with explicit hops.

// RUN: air-translate --air-to-spada %s | FileCheck %s

// CHECK: kernel @tree_reduce<>(stream<f32, 8>[8, 1] readonly arg0_in,
// CHECK: stream<f32> t0_even = relative_stream(-1, 0) { hops = [(-1, 0)], channel = 0 }
// CHECK: stream<f32> t1_even = relative_stream(-2, 0) { hops = [(-1, 0), (-1, 0)], channel = 2 }
// CHECK: stream<f32> t2_odd = relative_stream(-4, 0) { hops = [(-1, 0), (-1, 0), (-1, 0), (-1, 0)], channel = 5 }
// CHECK: compute i16 x, i16 y in [1:8:2, 0] {
// CHECK: compute i16 x, i16 y in [2:7:4, 0] {
// CHECK: await send(tree_l1_0, t2_odd)
// CHECK: await send(l2_1, arg1_out[x, y])

#map = affine_map<(d0, d1) -> (d0, 0)>
#map1 = affine_map<(d0) -> (0, 0)>
#map2 = affine_map<(d0) -> (d0 - 1)>
#map3 = affine_map<(d0) -> (d0 - 2)>
#map4 = affine_map<(d0) -> (d0 - 4)>
#set = affine_set<(d0, d1) : (d0 mod 2 - 1 == 0)>
#set1 = affine_set<(d0, d1) : (d0 mod 2 == 0)>
#set2 = affine_set<(d0, d1) : (d0 mod 4 - 2 == 0)>
#set3 = affine_set<(d0, d1) : (d0 mod 4 == 0)>
#set4 = affine_set<(d0, d1) : (d0 - 4 == 0)>
#set5 = affine_set<(d0, d1) : (d0 == 0)>
module {
  air.channel @t0 [8, 1]
  air.channel @t1 [8, 1]
  air.channel @t2 [8, 1]
  func.func @tree_reduce(%arg0: memref<8x8xf32>, %arg1: memref<8xf32>) {
    %c1 = arith.constant 1 : index
    air.launch (%arg2) in (%arg3=%c1) args(%arg4=%arg0, %arg5=%arg1) : memref<8x8xf32>, memref<8xf32> {
      air.segment @seg  args(%arg6=%arg4, %arg7=%arg5) : memref<8x8xf32>, memref<8xf32> {
        %c8 = arith.constant 8 : index
        %c1_0 = arith.constant 1 : index
        %alloc = memref.alloc() {air.partition = #air.partition<block = [1, 8], owner = #map>} : memref<8x8xf32, 1>
        %alloc_1 = memref.alloc() {air.partition = #air.partition<block = [8], owner = #map1>} : memref<8xf32, 1>
        air.dma_memcpy_nd (%alloc[] [] [], %arg6[] [] []) : (memref<8x8xf32, 1>, memref<8x8xf32>)
        air.herd @tree  tile (%arg8, %arg9) in (%arg10=%c8, %arg11=%c1_0) args(%arg12=%alloc, %arg13=%alloc_1) : memref<8x8xf32, 1>, memref<8xf32, 1> attributes {x_loc = 0 : i64, y_loc = 0 : i64} {
          %c0 = arith.constant 0 : index
          %c1_2 = arith.constant 1 : index
          %c8_3 = arith.constant 8 : index
          %alloc_4 = memref.alloc() : memref<8xf32, 2>
          %alloc_5 = memref.alloc() : memref<8xf32, 2>
          air.dma_memcpy_nd (%alloc_4[] [] [], %arg12[%arg8, %c0] [%c1_2, %c8_3] [%c8_3, %c1_2]) : (memref<8xf32, 2>, memref<8x8xf32, 1>)
          %0 = affine.apply #map2(%arg8)
          %1 = affine.apply #map3(%arg8)
          %2 = affine.apply #map4(%arg8)
          affine.if #set(%arg8, %arg9) {
            air.channel.put  @t0[%0, %arg9] (%alloc_4[] [] []) : (memref<8xf32, 2>)
          }
          affine.if #set1(%arg8, %arg9) {
            air.channel.get  @t0[%arg8, %arg9] (%alloc_5[] [] []) : (memref<8xf32, 2>)
            affine.for %arg14 = 0 to 8 {
              %3 = affine.load %alloc_4[%arg14] : memref<8xf32, 2>
              %4 = affine.load %alloc_5[%arg14] : memref<8xf32, 2>
              %5 = arith.addf %3, %4 : f32
              affine.store %5, %alloc_4[%arg14] : memref<8xf32, 2>
            }
          }
          affine.if #set2(%arg8, %arg9) {
            air.channel.put  @t1[%1, %arg9] (%alloc_4[] [] []) : (memref<8xf32, 2>)
          }
          affine.if #set3(%arg8, %arg9) {
            air.channel.get  @t1[%arg8, %arg9] (%alloc_5[] [] []) : (memref<8xf32, 2>)
            affine.for %arg14 = 0 to 8 {
              %3 = affine.load %alloc_4[%arg14] : memref<8xf32, 2>
              %4 = affine.load %alloc_5[%arg14] : memref<8xf32, 2>
              %5 = arith.addf %3, %4 : f32
              affine.store %5, %alloc_4[%arg14] : memref<8xf32, 2>
            }
          }
          affine.if #set4(%arg8, %arg9) {
            air.channel.put  @t2[%2, %arg9] (%alloc_4[] [] []) : (memref<8xf32, 2>)
          }
          affine.if #set5(%arg8, %arg9) {
            air.channel.get  @t2[%arg8, %arg9] (%alloc_5[] [] []) : (memref<8xf32, 2>)
            affine.for %arg14 = 0 to 8 {
              %3 = affine.load %alloc_4[%arg14] : memref<8xf32, 2>
              %4 = affine.load %alloc_5[%arg14] : memref<8xf32, 2>
              %5 = arith.addf %3, %4 : f32
              affine.store %5, %alloc_4[%arg14] : memref<8xf32, 2>
            }
            air.dma_memcpy_nd (%arg13[%c0] [%c8_3] [%c1_2], %alloc_4[] [] []) : (memref<8xf32, 1>, memref<8xf32, 2>)
          }
          memref.dealloc %alloc_4 : memref<8xf32, 2>
          memref.dealloc %alloc_5 : memref<8xf32, 2>
        }
        air.dma_memcpy_nd (%arg7[] [] [], %alloc_1[] [] []) : (memref<8xf32>, memref<8xf32, 1>)
        memref.dealloc %alloc : memref<8x8xf32, 1>
        memref.dealloc %alloc_1 : memref<8xf32, 1>
      }
    }
    return
  }
}

