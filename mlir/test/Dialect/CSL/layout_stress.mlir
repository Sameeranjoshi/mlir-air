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

  %r_east = csl.route in(RAMP) out(EAST) : !csl.route
  %r_west = csl.route in(RAMP) out(EAST) : !csl.route
  %r_north = csl.route in(RAMP) out(EAST) : !csl.route
  %r_south = csl.route in(RAMP) out(EAST) : !csl.route

  // A microkernel that takes multiple parameters (dictionary)
  %k_complex = csl.kernel {
  } {source_file = "complex_pe.csl"} : !csl.kernel

  // Huge region with many routes and colors
  %region_huge = csl.code_region routes(%r_east, %r_west, %r_north, %r_south) colors(%c_red, %c_blue, %c_green) {
  } {width = 100 : i64, height = 100 : i64} : !csl.code_region

  csl.place %region_huge %k_complex {x = 0 : i64, y = 0 : i64}

  %region_sink = csl.code_region routes(%r_west, %r_north) colors(%c_red, %c_green) {
  } {width = 10 : i64, height = 10 : i64} : !csl.code_region

  csl.place %region_sink %k_complex {x = 100 : i64, y = 100 : i64}
}

// 2. What happens if a paint operation specifies coordinates outside the region shape?
// Example: Region is 2x2, but we paint at (5, 5).
// Currently, our dialect lacks an MLIR verifier to catch this OOB access.
// CHECK: csl.spatial_placement {
csl.spatial_placement {
  %c = csl.color : !csl.color
  %r = csl.route in(RAMP) out(EAST) : !csl.route
  %reg_oob = csl.code_region routes(%r) colors(%c) {
  } {width = 2 : i64, height = 2 : i64} : !csl.code_region
}

// 3. What happens if multiple regions are placed at overlapping coordinates?
// Example: Region1 at (0,0) with shape 10x10. Region2 at (5,5).
// We have no global verification of placement collisions yet.
// CHECK: csl.spatial_placement {
csl.spatial_placement {
  %c = csl.color : !csl.color
  %r = csl.route in(RAMP) out(EAST) : !csl.route
  %reg1 = csl.code_region routes(%r) colors(%c) { } {width = 10 : i64, height = 10 : i64} : !csl.code_region
  %reg2 = csl.code_region routes(%r) colors(%c) { } {width = 10 : i64, height = 10 : i64} : !csl.code_region

  %k = csl.kernel {
  } {source_file = "pe.csl"} : !csl.kernel

  csl.place %reg1 %k {x = 0 : i64, y = 0 : i64}
  csl.place %reg2 %k {x = 5 : i64, y = 5 : i64} // Overlaps!
}
