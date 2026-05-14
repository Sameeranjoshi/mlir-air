//===- runtime_ops.mlir - CSL runtime dialect ops tests --------*- MLIR -*-===//
//
// Part of the air-to-csl project.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
//
// Tests for csl.import_module (stable runtime op).
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s | FileCheck %s

// CHECK-LABEL: func.func @main

func.func @main() {
  // CHECK: csl.import_module "<memcpy/memcpy>" : !csl.imported_module
  %mod = csl.import_module "<memcpy/memcpy>" : !csl.imported_module

  // CHECK: csl.import_module "<memcpy/get_params>" {params = {height = 2 : i32, width = 2 : i32}} : !csl.imported_module
  %mod2 = csl.import_module "<memcpy/get_params>" {params = {width = 2 : i32, height = 2 : i32}} : !csl.imported_module

  // CHECK: csl.import_module "<math>" : !csl.imported_module
  %mod3 = csl.import_module "<math>" : !csl.imported_module

  return
}

