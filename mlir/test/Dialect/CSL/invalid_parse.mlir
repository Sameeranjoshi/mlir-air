//===- invalid_parse.mlir - CSL parser rejection tests ---------*- MLIR -*-===//
//
// Part of the air-to-csl project.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt --verify-diagnostics --split-input-file %s

// -----

// Missing color keyword in task
// expected-error @+1 {{expected 'color'}}
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
