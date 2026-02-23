//===- layout_ops.mlir - CSL layout dialect ops tests ----------*- MLIR -*-===//
//
// Part of the air-to-csl project.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s | FileCheck %s

// CHECK-LABEL: module {

// Minimal 2x2 grid
// CHECK: csl.layout {
// CHECK:   csl.set_rectangle 2, 2
// CHECK: }
csl.layout {
  csl.set_rectangle 2, 2
}

// Non-square grid
// CHECK: csl.layout {
// CHECK:   csl.set_rectangle 4, 8
// CHECK: }
csl.layout {
  csl.set_rectangle 4, 8
}

// WSE-3 max dimensions
// CHECK: csl.layout {
// CHECK:   csl.set_rectangle 750, 994
// CHECK: }
csl.layout {
  csl.set_rectangle 750, 994
}

// Single PE
// CHECK: csl.layout {
// CHECK:   csl.set_rectangle 1, 1
// CHECK: }
csl.layout {
  csl.set_rectangle 1, 1
}

// Layout with multiple ops inside
// CHECK: csl.layout {
// CHECK:   csl.set_rectangle 3, 3
// CHECK:   csl.set_tile_code 0, 0 file("pe.csl")
// CHECK:   csl.export_name "data" : f32
// CHECK: }
csl.layout {
  csl.set_rectangle 3, 3
  csl.set_tile_code 0, 0 file("pe.csl")
  csl.export_name "data" : f32
}
