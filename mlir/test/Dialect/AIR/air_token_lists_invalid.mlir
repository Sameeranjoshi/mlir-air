//===- air_token_lists_invalid.mlir ----------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -split-input-file -verify-diagnostics

func.func @alloc_as_dependency() {
  %c1 = arith.constant 1 : index
  // expected-error@+1 {{result is used as an async dependency}}
  %t = air.token.alloc : !air.async.token
  %d = air.herd async [%t] tile (%x, %y) in (%sx = %c1, %sy = %c1) {
    air.herd_terminator
  }
  return
}

// -----

func.func @token_crosses_launch() {
  %c1 = arith.constant 1 : index
  %t = air.token.alloc : !air.async.token
  air.launch (%lx) in (%lsx = %c1) args(%lt = %t) : !air.async.token {
    // expected-error@+1 {{may not cross the launch boundary}}
    air.segment @seg affinity [%lt] {
      air.segment_terminator
    }
    air.launch_terminator
  }
  return
}

// -----

func.func @launch_concurrency() {
  %c1 = arith.constant 1 : index
  %t = air.token.alloc : !air.async.token
  // expected-error@+1 {{air.launch may not carry a concurrency list}}
  air.launch concurrency [%t] (%lx) in (%lsx = %c1) {
    air.launch_terminator
  }
  return
}
