//===- air_token_lists.mlir ------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
// Affinity / concurrency token lists (AIRComputeModel.md §1.5) round-trip.

// RUN: air-opt %s | air-opt | FileCheck %s

// CHECK-LABEL: func.func @token_lists
// CHECK: %[[T:.*]] = air.token.alloc : !air.async.token
// CHECK: %[[C:.*]] = air.token.alloc : !air.async.token
// CHECK: air.launch affinity [%[[T]]] (%{{.*}}) in (%{{.*}}=%c1)
// CHECK: %[[TA:.*]] = air.token.alloc
// CHECK: %[[TC:.*]] = air.token.alloc
// CHECK: air.segment @seg affinity [%[[TA]]] concurrency [%[[TC]]]
// CHECK: %[[HA:.*]] = air.token.alloc
// CHECK: %[[HC:.*]] = air.token.alloc
// CHECK: air.herd @h0 affinity [%[[HA]]] concurrency [%[[HC]]] tile
// CHECK: %[[D:.*]] = air.herd @h1 async affinity [%[[HA]]] tile
// CHECK: air.herd @h2 async [%[[D]]] affinity [%[[HA]]] tile
func.func @token_lists() {
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %t = air.token.alloc : !air.async.token
  %c = air.token.alloc : !air.async.token
  air.launch affinity [%t] (%lx) in (%lsx = %c1) {
    %ta = air.token.alloc : !air.async.token
    %tc = air.token.alloc : !air.async.token
    air.segment @seg affinity [%ta] concurrency [%tc] {
      %c2_0 = arith.constant 2 : index
      %ta2 = air.token.alloc : !air.async.token
      %tc2 = air.token.alloc : !air.async.token
      air.herd @h0 affinity [%ta2] concurrency [%tc2] tile (%x, %y) in (%sx = %c2_0, %sy = %c2_0) {
        air.herd_terminator
      }
      %d = air.herd @h1 async affinity [%ta2] tile (%x, %y) in (%sx = %c2_0, %sy = %c2_0) {
        air.herd_terminator
      }
      %e = air.herd @h2 async [%d] affinity [%ta2] tile (%x, %y) in (%sx = %c2_0, %sy = %c2_0) {
        air.herd_terminator
      }
      air.segment_terminator
    }
    air.launch_terminator
  }
  return
}
