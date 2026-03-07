//===- emit.mlir - CSL Runtime translation tests -----------*- MLIR -*-===//
//
// Part of the air-to-csl project.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
//
// Tests for air-translate --emit-csl-rt: CSL → CSL Runtime → Python files.
//
// Pipeline: CSL dialect with spatial_placement
//           → (-csl-to-csl-rt pass)
//           → CSL Runtime dialect
//           → (--emit-csl-rt)
//           → layout.py + run.py + kernel.csl
//
//===----------------------------------------------------------------------===//

// ============================================================================
// Test: CSL spatial_placement -> CSL Runtime -> Python
//   Verifies: module, param, var, func, task, comptime, export_symbol (PE level)
//            spatial_placement, code_region, paint, place (converted to CSL Runtime)
//
// RUN: air-opt --csl-to-csl-rt %s | air-translate --emit-csl-rt --csl-output-dir=%t
// TODO: Add FileCheck patterns for generated layout.py and kernel.csl
// ============================================================================
%k = csl.kernel "pe_program.csl" params({memcpy_params = 0 : i64}) {
  // Data in memory(can be own or borrowed memory from neighbor node.)
  csl.var @arg_0 : memref<1024xf32>
  csl.var @arg_1 : memref<1024xf32>

  // Function with arithmetic body (IsolatedFromAbove, uses local ops only)
  csl.func @compute() : () -> () {
    %c0   = arith.constant 0 : i32
    %a    = arith.constant 1.5 : f32
    %b    = arith.constant 2.5 : f32
    %sum  = arith.addf %a, %b : f32
    csl.return
  }

  csl.func @init_and_compute() : () -> () {
    csl.return
  }

  csl.comptime {
    csl.export_symbol @arg_0 alias("arg_0")
    csl.export_symbol @arg_1 alias("arg_1")
    csl.export_symbol @init_and_compute
  }
} : !csl.kernel

csl.spatial_placement {
  // resources for hardware, can assume infinite, compiler must be taking care of their allocation.
  %c = csl.color : !csl.color
  %r = csl.route in(RAMP) out(EAST) : i32
  %region = csl.code_region routes(%r) colors(%c) shape(2, 2) {
    csl.paint pe(0, 0) route(%r) color(%c)
  } : !csl.code_region

  csl.place %region at(0, 0) kernel(%k)
}
