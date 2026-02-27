//===- layout_stress.mlir - CSL layout ops edge cases/stress tests -*- MLIR -*-===//
//
// Part of the air-to-csl project.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
//
// This file tests edge cases, complex combinations, and potential flaws
// in the current CSL layout op design.
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s | FileCheck %s

// CHECK-LABEL: module {

// 1. Extreme Combinations: Many regions, kernels, shared colors and routes
// CHECK: csl.spatial_placement {
csl.spatial_placement {
  %c_red = csl.color : !csl.color
  %c_blue = csl.color : !csl.color
  %c_green = csl.color : !csl.color

  %r_east = csl.route in(RAMP) out(EAST) : i32
  %r_west = csl.route in(RAMP) out(WEST) : i32
  %r_north = csl.route in(RAMP) out(NORTH) : i32
  %r_south = csl.route in(RAMP) out(SOUTH) : i32

  // A microkernel that takes multiple parameters (dictionary)
  %k_complex = csl.kernel @complex_pe ({tile_id = 5 : i32, max_iters = 100 : i32, is_master = true}) {
    csl.func @run() { csl.return }
  } : !csl.kernel

  // Huge region with many routes and colors
  %region_huge = csl.code_region routes(%r_east, %r_west, %r_north, %r_south) colors(%c_red, %c_blue, %c_green) shape(100, 100) {
    // Sparse painting (not every PE painted)
    csl.paint pe(0, 0) route(%r_east) color(%c_red)
    csl.paint pe(99, 99) route(%r_west) color(%c_blue)
    csl.paint pe(50, 50) route(%r_north) color(%c_green)
  } : !csl.code_region

  csl.place %region_huge at(0, 0) kernel(%k_complex)

  // Multiple ports on the same physical edge of the same region
  %port_out1 = csl.port %region_huge type("output") route(%r_east) size(1024) : !csl.port
  %port_out2 = csl.port %region_huge type("output") route(%r_south) size(512) : !csl.port
  
  %region_sink = csl.code_region routes(%r_west, %r_north) colors(%c_red, %c_green) shape(10, 10) {
  } : !csl.code_region
  
  csl.place %region_sink at(100, 100) kernel(%k_complex)
  
  %port_in1 = csl.port %region_sink type("input") route(%r_west) size(1024) : !csl.port
  %port_in2 = csl.port %region_sink type("input") route(%r_north) size(512) : !csl.port

  // Forked dataflow (one output to multiple inputs) -> Does our MLIR allow this semantically?
  // Currently, yes, because SSA values aren't consumed.
  csl.dataflow %port_out1 -> %port_in1
  csl.dataflow %port_out2 -> %port_in2
}

// 2. What happens if a paint operation specifies coordinates outside the region shape?
// Example: Region is 2x2, but we paint at (5, 5). 
// Currently, our dialect lacks an MLIR verifier to catch this OOB access.
// CHECK: csl.spatial_placement {
csl.spatial_placement {
  %c = csl.color : !csl.color
  %r = csl.route in(RAMP) out(EAST) : i32
  %reg_oob = csl.code_region routes(%r) colors(%c) shape(2, 2) {
    csl.paint pe(5, 5) route(%r) color(%c) 
  } : !csl.code_region
}

// 3. What happens if multiple regions are placed at overlapping coordinates?
// Example: Region1 at (0,0) with shape 10x10. Region2 at (5,5).
// We have no global verification of placement collisions yet.
// CHECK: csl.spatial_placement {
csl.spatial_placement {
  %c = csl.color : !csl.color
  %r = csl.route in(RAMP) out(EAST) : i32
  %reg1 = csl.code_region routes(%r) colors(%c) shape(10, 10) { } : !csl.code_region
  %reg2 = csl.code_region routes(%r) colors(%c) shape(10, 10) { } : !csl.code_region
  
  %k = csl.kernel "pe.csl" { } : !csl.kernel

  csl.place %reg1 at(0, 0) kernel(%k)
  csl.place %reg2 at(5, 5) kernel(%k) // Overlaps!
}
