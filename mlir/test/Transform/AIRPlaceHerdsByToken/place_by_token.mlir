//===- place_by_token.mlir -------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-place-herds-by-token="fabric-x=4 fabric-y=4" | FileCheck %s

#part = #air.partition<block = [8], owner = affine_map<(i) -> (i, 0)>>
#row0 = affine_set<(x, y) : (y == 0)>

air.channel @c [2, 1]

// Two herds with the same affinity token land on the same tiles; the third,
// unrelated herd is placed on a disjoint rectangle. A producer/consumer pair
// connected by a channel is disjoint as well.
// CHECK-LABEL: func.func @placement
// CHECK: air.herd @a affinity {{.*}} attributes {x_loc = 0 : i64, y_loc = 0 : i64}
// CHECK: air.herd @b affinity {{.*}} attributes {x_loc = 0 : i64, y_loc = 0 : i64}
// CHECK: air.herd @c tile {{.*}} attributes {x_loc = 2 : i64, y_loc = 0 : i64}
// CHECK: air.herd @prod tile {{.*}} attributes {x_loc = 0 : i64, y_loc = 2 : i64}
// CHECK: air.herd @cons tile {{.*}} attributes {x_loc = 2 : i64, y_loc = 2 : i64}
func.func @placement() {
  %c1 = arith.constant 1 : index
  air.launch (%lx) in (%lsx = %c1) {
    air.segment @seg {
      %c2 = arith.constant 2 : index
      %c1_0 = arith.constant 1 : index
      %t = air.token.alloc : !air.async.token
      air.herd @a affinity [%t] tile (%x, %y) in (%sx = %c2, %sy = %c2) {
        air.herd_terminator
      }
      air.herd @b affinity [%t] tile (%x, %y) in (%sx = %c2, %sy = %c2) {
        air.herd_terminator
      }
      air.herd @c tile (%x, %y) in (%sx = %c2, %sy = %c2) {
        air.herd_terminator
      }
      air.herd @prod tile (%x, %y) in (%sx = %c2, %sy = %c1_0) {
        %buf = memref.alloc() : memref<8xf32, 2>
        air.channel.put @c[%x, %y] (%buf[] [] []) : (memref<8xf32, 2>)
        memref.dealloc %buf : memref<8xf32, 2>
        air.herd_terminator
      }
      air.herd @cons tile (%x, %y) in (%sx = %c2, %sy = %c1_0) {
        %buf = memref.alloc() : memref<8xf32, 2>
        air.channel.get @c[%x, %y] (%buf[] [] []) : (memref<8xf32, 2>)
        memref.dealloc %buf : memref<8xf32, 2>
        air.herd_terminator
      }
      air.segment_terminator
    }
    air.launch_terminator
  }
  return
}

// Owner-local access to a partitioned L2 memref, guarded by an affine.if on
// the tile ids, is accepted.
// CHECK-LABEL: func.func @owned_access
// CHECK: air.herd @h
func.func @owned_access() {
  %c1 = arith.constant 1 : index
  air.launch (%lx) in (%lsx = %c1) {
    air.segment @seg {
      %c2 = arith.constant 2 : index
      %x2 = memref.alloc() {air.partition = #part} : memref<16xf32, 1>
      air.herd @h tile (%x, %y) in (%sx = %c2, %sy = %c2) args(%hx = %x2) : memref<16xf32, 1> {
        %c1_h = arith.constant 1 : index
        %c8_h = arith.constant 8 : index
        %l = memref.alloc() : memref<8xf32, 2>
        %off = affine.apply affine_map<(d0) -> (d0 * 8)>(%x)
        affine.if #row0(%x, %y) {
          air.dma_memcpy_nd (%l[] [] [], %hx[%off] [%c8_h] [%c1_h]) : (memref<8xf32, 2>, memref<16xf32, 1>)
        }
        memref.dealloc %l : memref<8xf32, 2>
        air.herd_terminator
      }
      memref.dealloc %x2 : memref<16xf32, 1>
      air.segment_terminator
    }
    air.launch_terminator
  }
  return
}
