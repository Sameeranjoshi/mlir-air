//===- invalid.mlir - CSL dialect negative/diagnostic tests ----*- MLIR -*-===//
//
// Part of the air-to-csl project.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt --verify-diagnostics --split-input-file %s

// -----

// csl.return outside of csl.func or csl.task
func.func @bad_return() {
  // expected-error @+1 {{'csl.return' op expects parent op to be one of 'csl.func, csl.task'}}
  csl.return
}

// -----

// csl.return inside a bare module (not func/task)
module {
  // expected-error @+1 {{'csl.return' op expects parent op to be one of 'csl.func, csl.task'}}
  csl.return
}

// -----

// csl.var with Symbol trait requires parent with SymbolTable
csl.spatial_placement {
  // expected-error @+1 {{'csl.var' op symbol's parent must have the SymbolTable trait}}
  csl.var @buf : memref<1024xf32>
}

// -----

// csl.param with Symbol trait requires parent with SymbolTable
csl.spatial_placement {
  // expected-error @+1 {{'csl.param' op symbol's parent must have the SymbolTable trait}}
  csl.param @tile_id : i32
}

// -----

// csl.func with Symbol trait requires parent with SymbolTable
csl.spatial_placement {
  // expected-error @+1 {{'csl.func' op symbol's parent must have the SymbolTable trait}}
  csl.func @compute() {
    csl.return
  }
}
