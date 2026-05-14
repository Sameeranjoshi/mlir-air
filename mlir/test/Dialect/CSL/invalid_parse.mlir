//===- invalid_parse.mlir - CSL parser rejection tests ---------*- MLIR -*-===//
//
// Part of the air-to-csl project.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt --verify-diagnostics --split-input-file %s

// -----

// Stray '(' after task symbol: task is now attribute-driven and has no
// operand list, so the parser expects either `attributes {...}` or the
// region body.
// expected-error @+1 {{expected '{' to begin a region}}
csl.task @bad_task() : () -> () {
  csl.return
}

// -----

// Missing params keyword in import_module (just a stray dict)
// expected-error @+1 {{expected ':'}}
%m = csl.import_module "<mod>" ({key = 1 : i32}) : !csl.imported_module

// -----

// Invalid direction enum in route
// expected-error @below {{expected string or keyword containing one of the following enum values for attribute 'input_dir'}}
%r = csl.route in(UP) out(WEST) : !csl.route
