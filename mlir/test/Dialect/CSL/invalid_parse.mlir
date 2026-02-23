//===- invalid_parse.mlir - CSL parser rejection tests ---------*- MLIR -*-===//
//
// Part of the air-to-csl project.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
//
// Tests that the CSL dialect parser rejects syntactically malformed input.
//
//===----------------------------------------------------------------------===//

// RUN: air-opt --verify-diagnostics --split-input-file %s

// -----

// Missing file keyword in set_tile_code
csl.layout {
  csl.set_rectangle 2, 2
  // expected-error @+1 {{expected 'file'}}
  csl.set_tile_code 0, 0 ("pe_program.csl")
}

// -----

// Missing color keyword in task
// expected-error @+1 {{expected 'color'}}
csl.task @bad_task() {
  csl.return
}

// -----

// Missing params keyword in import_module (just a stray dict)
// expected-error @+1 {{expected ':'}}
%m = csl.import_module "<mod>" ({key = 1 : i32}) : !csl.imported_module

// -----

// Invalid direction enum in route
%c = csl.color 0 : !csl.color
// expected-error @below {{custom op 'csl.route' expected string or keyword containing one of the following enum values for attribute 'direction' [NORTH, SOUTH, EAST, WEST, RAMP]}}
csl.route %c dir(UP)
