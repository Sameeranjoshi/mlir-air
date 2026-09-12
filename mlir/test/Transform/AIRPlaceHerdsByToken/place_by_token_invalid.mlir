//===- place_by_token_invalid.mlir -----------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -split-input-file -air-place-herds-by-token="fabric-x=4 fabric-y=4" -verify-diagnostics

#part = #air.partition<block = [8], owner = affine_map<(i) -> (i, 0)>>

// Row 1 tiles read x blocks owned by row 0: not an owner access.
func.func @non_owner_access() {
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
        // expected-error@+1 {{tile (0, 1) accesses element 0 of a partitioned L2 memref owned by tile (0, 0); non-owner access must use an air.channel}}
        air.dma_memcpy_nd (%l[] [] [], %hx[%off] [%c8_h] [%c1_h]) : (memref<8xf32, 2>, memref<16xf32, 1>)
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

// -----

air.channel @c [2, 2]

// A channel between two herds of one affinity class can never make progress.
func.func @affinity_channel_deadlock() {
  %c1 = arith.constant 1 : index
  air.launch (%lx) in (%lsx = %c1) {
    air.segment @seg {
      %c2 = arith.constant 2 : index
      %t = air.token.alloc : !air.async.token
      air.herd @prod affinity [%t] tile (%x, %y) in (%sx = %c2, %sy = %c2) {
        %buf = memref.alloc() : memref<8xf32, 2>
        air.channel.put @c[%x, %y] (%buf[] [] []) : (memref<8xf32, 2>)
        memref.dealloc %buf : memref<8xf32, 2>
        air.herd_terminator
      }
      // expected-error@+1 {{communicates through channel @c with a herd in the same affinity class}}
      air.herd @cons affinity [%t] tile (%x, %y) in (%sx = %c2, %sy = %c2) {
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

// -----

func.func @affinity_shape_mismatch() {
  %c1 = arith.constant 1 : index
  air.launch (%lx) in (%lsx = %c1) {
    air.segment @seg {
      %c2 = arith.constant 2 : index
      %c1_s = arith.constant 1 : index
      %t = air.token.alloc : !air.async.token
      air.herd @a affinity [%t] tile (%x, %y) in (%sx = %c2, %sy = %c2) {
        air.herd_terminator
      }
      // expected-error@+1 {{shares an affinity token with a herd of a different shape}}
      air.herd @b affinity [%t] tile (%x, %y) in (%sx = %c2, %sy = %c1_s) {
        air.herd_terminator
      }
      air.segment_terminator
    }
    air.launch_terminator
  }
  return
}
