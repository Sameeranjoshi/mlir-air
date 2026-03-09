//===- kernel_ops.mlir - CSL kernel dialect ops tests ----------*- MLIR -*-===//
//
// Part of the air-to-csl project.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s | FileCheck %s

// CHECK-LABEL: module {

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

// Task bound to a color
// CHECK: csl.task @recv_task color(%{{.*}}) {
// CHECK:   csl.return
// CHECK: }
%c3 = csl.color {id = 3 : i32} : !csl.color
csl.task @recv_task color(%c3) {
  csl.return
}

// Task bound to color 0
// CHECK: csl.task @send_task color(%{{.*}}) {
// CHECK:   csl.return
// CHECK: }
%c0 = csl.color {id = 0 : i32} : !csl.color
csl.task @send_task color(%c0) {
  csl.return
}

// Task with higher color id
// CHECK: csl.task @data_task color(%{{.*}}) {
// CHECK:   csl.return
// CHECK: }
%c15 = csl.color {id = 15 : i32} : !csl.color
csl.task @data_task color(%c15) {
  csl.return
}
