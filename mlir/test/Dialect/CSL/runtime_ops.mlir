//===- runtime_ops.mlir - CSL runtime dialect ops tests --------*- MLIR -*-===//
//
// Part of the air-to-csl project.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
//
// Tests for CSL ops that were consolidated or removed.
// NOTE: csl.param, csl.export_symbol, and csl.comptime were removed in the
// dialect consolidation (2026-03-08). They are now only available via the
// CSL Runtime dialect (csl_rt.*) for higher-level SDK abstractions.
// See MEMORY.md for consolidation details.
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

// NOTE: These ops were removed in the consolidation (2026-03-08):
// - csl.param (now only csl_rt.set_param_all in CSL Runtime dialect)
// - csl.export_symbol (now only csl_rt.export_name in CSL Runtime dialect)
// - csl.comptime (inferred by compiler passes)
//
// Use csl_rt.* ops (CSL Runtime dialect) for parameter and export operations.
