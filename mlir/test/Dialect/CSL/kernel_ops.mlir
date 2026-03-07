//===- kernel_ops.mlir - CSL kernel dialect ops tests ----------*- MLIR -*-===//
//
// Part of the air-to-csl project.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s | FileCheck %s

// CHECK-LABEL: module {

// Variable declarations with different types and ranks
// CHECK: csl.var @buf : memref<1024xf32>
// CHECK: csl.var @matrix : memref<32x32xf16>
// CHECK: csl.var @scalar : f32
// CHECK: csl.var @vector_i16 : memref<256xi16>
csl.var @buf : memref<1024xf32>
csl.var @matrix : memref<32x32xf16>
csl.var @scalar : f32
csl.var @vector_i16 : memref<256xi16>

// Empty function
// CHECK: csl.func @noop() : () -> () {
// CHECK:   csl.return
// CHECK: }
csl.func @noop() : () -> () {
  csl.return
}

// Function with compute body
// CHECK: csl.func @compute() : () -> () {
// CHECK:   csl.return
// CHECK: }
csl.func @compute() : () -> () {
  csl.return
}

// Multiple functions
// CHECK: csl.func @init() : () -> () {
// CHECK:   csl.return
// CHECK: }
// CHECK: csl.func @finalize() : () -> () {
// CHECK:   csl.return
// CHECK: }
csl.func @init() : () -> () {
  csl.return
}
csl.func @finalize() : () -> () {
  csl.return
}

// Task bound to a color
// CHECK: csl.task @recv_task() color(3) : () -> () {
// CHECK:   csl.return
// CHECK: }
csl.task @recv_task() color(3) : () -> () {
  csl.return
}

// Task bound to color 0
// CHECK: csl.task @send_task() color(0) : () -> () {
// CHECK:   csl.return
// CHECK: }
csl.task @send_task() color(0) : () -> () {
  csl.return
}

// Task with higher color id
// CHECK: csl.task @data_task() color(15) : () -> () {
// CHECK:   csl.return
// CHECK: }
csl.task @data_task() color(15) : () -> () {
  csl.return
}
