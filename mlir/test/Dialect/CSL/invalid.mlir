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

// csl.func with Symbol trait requires parent with SymbolTable (must be in csl.kernel)
csl.spatial_placement {
  // expected-error @+1 {{'csl.func' op symbol's parent must have the SymbolTable trait}}
  csl.func @compute {
    csl.return
  }
}
