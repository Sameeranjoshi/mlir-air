//===- datamovement_ops.mlir - CSL data movement ops tests ----*- MLIR -*-===//
//
// Part of the air-to-csl project.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s | FileCheck %s

// CHECK-LABEL: func.func @test_mem_dsd
func.func @test_mem_dsd(%buf : memref<1024xf32>) {
  %len = arith.constant 1024 : index
  // CHECK: %[[DSD:.*]] = csl.get_mem_dsd %{{.*}}, %{{.*}} : memref<1024xf32>, index -> !csl.dsd
  // future layout = [stride, offset, length, ...]
  // %m = memref<100xf32, which memory space?>
  // memref<?xf32> = layout.apply(futurelayout, %m)
  %dsd = csl.get_mem_dsd %buf, %len : memref<1024xf32>, index -> !csl.dsd
  return
}

// CHECK-LABEL: func.func @test_mem_dsd_f16
func.func @test_mem_dsd_f16(%buf : memref<512xf16>) {
  %len = arith.constant 512 : index
  // CHECK: csl.get_mem_dsd %{{.*}}, %{{.*}} : memref<512xf16>, index -> !csl.dsd
  %dsd = csl.get_mem_dsd %buf, %len : memref<512xf16>, index -> !csl.dsd
  return
}

// CHECK-LABEL: func.func @test_fab_dsd
func.func @test_fab_dsd() {
  %len = arith.constant 256 : index
  // CHECK: %[[C:.*]] = csl.color 0 : !csl.color
  %c = csl.color 0 : !csl.color

  // CHECK: %[[IN:.*]] = csl.get_fab_dsd fabin %[[C]], %{{.*}} : !csl.color, index -> !csl.dsd
  %dsd_in = csl.get_fab_dsd fabin %c, %len : !csl.color, index -> !csl.dsd

  // CHECK: %[[OUT:.*]] = csl.get_fab_dsd fabout %[[C]], %{{.*}} : !csl.color, index -> !csl.dsd
  %dsd_out = csl.get_fab_dsd fabout %c, %len : !csl.color, index -> !csl.dsd
  return
}

// CHECK-LABEL: func.func @test_mov
func.func @test_mov(%buf : memref<1024xf32>) {
  %len = arith.constant 1024 : index
  %c = csl.color 0 : !csl.color

  %mem_dsd = csl.get_mem_dsd %buf, %len : memref<1024xf32>, index -> !csl.dsd
  %fab_in = csl.get_fab_dsd fabin %c, %len : !csl.color, index -> !csl.dsd
  %fab_out = csl.get_fab_dsd fabout %c, %len : !csl.color, index -> !csl.dsd

  // Receive from fabric into memory
  // CHECK: csl.mov %{{.*}}, %{{.*}} : !csl.dsd, !csl.dsd
  csl.mov %mem_dsd, %fab_in : !csl.dsd, !csl.dsd

  // Send from memory to fabric
  // CHECK: csl.mov %{{.*}}, %{{.*}} : !csl.dsd, !csl.dsd
  csl.mov %fab_out, %mem_dsd : !csl.dsd, !csl.dsd

  // Memory-to-memory copy
  // CHECK: csl.mov %{{.*}}, %{{.*}} : !csl.dsd, !csl.dsd
  csl.mov %mem_dsd, %mem_dsd : !csl.dsd, !csl.dsd

  return
}

// Multiple colors feeding into different DSDs
// CHECK-LABEL: func.func @test_multi_color_dsd
func.func @test_multi_color_dsd() {
  %len = arith.constant 128 : index
  // CHECK: %[[C0:.*]] = csl.color 0
  // CHECK: %[[C1:.*]] = csl.color 1
  %c0 = csl.color 0 : !csl.color
  %c1 = csl.color 1 : !csl.color

  // CHECK: csl.get_fab_dsd fabin %[[C0]]
  // CHECK: csl.get_fab_dsd fabout %[[C1]]
  %in0 = csl.get_fab_dsd fabin %c0, %len : !csl.color, index -> !csl.dsd
  %out1 = csl.get_fab_dsd fabout %c1, %len : !csl.color, index -> !csl.dsd

  csl.mov %out1, %in0 : !csl.dsd, !csl.dsd
  return
}
