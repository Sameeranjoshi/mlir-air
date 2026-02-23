//===- runtime_ops.mlir - CSL runtime dialect ops tests --------*- MLIR -*-===//
//
// Part of the air-to-csl project.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s | FileCheck %s

// CHECK-LABEL: module {

// Module import without params
// CHECK: %[[MOD:.*]] = csl.import_module "<memcpy/memcpy>" : !csl.imported_module
%mod = csl.import_module "<memcpy/memcpy>" : !csl.imported_module

// Module import with params
// CHECK: %[[MOD2:.*]] = csl.import_module "<memcpy/get_params>" params({height = 2 : i32, width = 2 : i32}) : !csl.imported_module
%mod2 = csl.import_module "<memcpy/get_params>" params({width = 2 : i32, height = 2 : i32}) : !csl.imported_module

// Multiple module imports
// CHECK: csl.import_module "<math>" : !csl.imported_module
%mod3 = csl.import_module "<math>" : !csl.imported_module

// Export name for data (layout-level host-visible symbols)
// CHECK: csl.export_name "data" : f32
// CHECK: csl.export_name "compute" : () -> ()
// CHECK: csl.export_name "buffer" : memref<1024xf32>
csl.export_name "data" : f32
csl.export_name "compute" : () -> ()
csl.export_name "buffer" : memref<1024xf32>

// Module parameters
// CHECK: csl.param @memcpy_params : i64
// CHECK: csl.param @tile_id : i32
// CHECK: csl.param @width : index
csl.param @memcpy_params : i64
csl.param @tile_id : i32
csl.param @width : index

// Export symbol without alias
// CHECK: csl.export_symbol @compute
csl.export_symbol @compute

// Export symbol with alias
// CHECK: csl.export_symbol @buf_ptr alias("data")
csl.export_symbol @buf_ptr alias("data")

// Comptime block with single export
// CHECK: csl.comptime {
// CHECK:   csl.export_symbol @compute
// CHECK: }
csl.comptime {
  csl.export_symbol @compute
}

// Comptime block with multiple exports
// CHECK: csl.comptime {
// CHECK:   csl.export_symbol @init alias("initialize")
// CHECK:   csl.export_symbol @buf alias("buffer")
// CHECK:   csl.export_symbol @finalize
// CHECK: }
csl.comptime {
  csl.export_symbol @init alias("initialize")
  csl.export_symbol @buf alias("buffer")
  csl.export_symbol @finalize
}
