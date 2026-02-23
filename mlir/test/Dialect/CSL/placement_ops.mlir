//===- placement_ops.mlir - CSL placement dialect ops tests ----*- MLIR -*-===//
//
// Part of the air-to-csl project.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s | FileCheck %s

// CHECK-LABEL: module {

// Basic tile placement without params
// CHECK: csl.layout {
// CHECK:   csl.set_rectangle 2, 2
// CHECK:   csl.set_tile_code 0, 0 file("pe_program.csl")
// CHECK:   csl.set_tile_code 1, 0 file("pe_program.csl")
// CHECK:   csl.set_tile_code 0, 1 file("pe_program.csl")
// CHECK:   csl.set_tile_code 1, 1 file("pe_program.csl")
// CHECK: }
csl.layout {
  csl.set_rectangle 2, 2
  csl.set_tile_code 0, 0 file("pe_program.csl")
  csl.set_tile_code 1, 0 file("pe_program.csl")
  csl.set_tile_code 0, 1 file("pe_program.csl")
  csl.set_tile_code 1, 1 file("pe_program.csl")
}

// Tile placement with compile-time params
// CHECK: csl.layout {
// CHECK:   csl.set_rectangle 1, 1
// CHECK:   csl.set_tile_code 0, 0 file("pe_program.csl") params({col = 0 : i32, row = 0 : i32})
// CHECK: }
csl.layout {
  csl.set_rectangle 1, 1
  csl.set_tile_code 0, 0 file("pe_program.csl") params({col = 0 : i32, row = 0 : i32})
}

// Different modules on different tiles
// CHECK: csl.layout {
// CHECK:   csl.set_rectangle 2, 1
// CHECK:   csl.set_tile_code 0, 0 file("sender.csl")
// CHECK:   csl.set_tile_code 1, 0 file("receiver.csl")
// CHECK: }
csl.layout {
  csl.set_rectangle 2, 1
  csl.set_tile_code 0, 0 file("sender.csl")
  csl.set_tile_code 1, 0 file("receiver.csl")
}

// Tile placement with multiple params
// CHECK: csl.layout {
// CHECK:   csl.set_rectangle 2, 2
// CHECK:   csl.set_tile_code 0, 0 file("pe.csl") params({col = 0 : i32, is_border = true, row = 0 : i32})
// CHECK: }
csl.layout {
  csl.set_rectangle 2, 2
  csl.set_tile_code 0, 0 file("pe.csl") params({col = 0 : i32, row = 0 : i32, is_border = true})
}
