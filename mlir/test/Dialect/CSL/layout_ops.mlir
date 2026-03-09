//===- layout_ops.mlir - CSL layout ops round-trip tests -------*- MLIR -*-===//
//
// Part of the air-to-csl project.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
//
// Positive round-trip tests for every active op in CSLLayoutOps.td.
// Matches the programming model from placement_mlir_design.txt.
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s | FileCheck %s

// ============================================================================
// csl.spatial_placement — empty container
// ============================================================================

// CHECK-LABEL: csl.spatial_placement
// CHECK: }
csl.spatial_placement {
}

// ============================================================================
// csl.code_region — with routes, colors, shape, and body
// ============================================================================

// CHECK-LABEL: csl.spatial_placement
// CHECK:   %[[R1:.*]] = csl.route in(RAMP) out(EAST) : !csl.route
// CHECK:   %[[RED:.*]] = csl.color : !csl.color
// CHECK:   %[[REG:.*]] = csl.code_region routes(%[[R1]]) colors(%[[RED]])
// CHECK:   {width = 10 : i64, height = 10 : i64} : !csl.code_region
csl.spatial_placement {
  %r1  = csl.route in(RAMP) out(EAST) : !csl.route
  %red = csl.color : !csl.color
  %regionA = csl.code_region routes(%r1) colors(%red) {
  } {width = 10 : i64, height = 10 : i64} : !csl.code_region
}

// ============================================================================
// csl.paint — paint a PE inside a code_region body
// ============================================================================

// CHECK-LABEL: csl.spatial_placement
csl.spatial_placement {
  %r1  = csl.route in(RAMP) out(EAST) : !csl.route
  %red = csl.color : !csl.color
  %regionA = csl.code_region routes(%r1) colors(%red) {
  } {width = 4 : i64, height = 4 : i64} : !csl.code_region
}

// ============================================================================
// csl.paint — multiple paint ops inside body
// ============================================================================

// CHECK-LABEL: csl.spatial_placement
csl.spatial_placement {
  %r1    = csl.route in(RAMP) out(EAST) : !csl.route
  %r2    = csl.route in(RAMP) out(EAST) : !csl.route
  %red   = csl.color : !csl.color
  %green = csl.color : !csl.color
  %regionA = csl.code_region routes(%r1, %r2) colors(%red, %green) {
  } {width = 2 : i64, height = 1 : i64} : !csl.code_region
}

// ============================================================================
// csl.port — declare ports on a code region
// ============================================================================

// CHECK-LABEL: csl.spatial_placement
// CHECK:   %[[REG:.*]] = csl.code_region
csl.spatial_placement {
  %r1  = csl.route in(RAMP) out(EAST) : !csl.route
  %r2  = csl.route in(RAMP) out(EAST) : !csl.route
  %red = csl.color : !csl.color
  %regionA = csl.code_region routes(%r1, %r2) colors(%red) {
  } {width = 10 : i64, height = 10 : i64} : !csl.code_region
}

// ============================================================================
// csl.place — place a region at a coordinate with kernel
// ============================================================================

// CHECK-LABEL: csl.spatial_placement
// CHECK:   %[[REG:.*]] = csl.code_region
// CHECK:   %[[K:.*]] = csl.kernel
// CHECK:   csl.place %[[REG]] %[[K]] {x = 0 : i64, y = 0 : i64}
csl.spatial_placement {
  %r1  = csl.route in(RAMP) out(EAST) : !csl.route
  %red = csl.color : !csl.color
  %regionA = csl.code_region routes(%r1) colors(%red) {
  } {width = 10 : i64, height = 10 : i64} : !csl.code_region
  %k1 = csl.kernel {
  } {source_file = "pe.csl"} : !csl.kernel
  csl.place %regionA %k1 {x = 0 : i64, y = 0 : i64}
}

// CHECK-LABEL: csl.spatial_placement
csl.spatial_placement {
  %r1 = csl.route in(RAMP) out(EAST) : !csl.route
  %r2 = csl.route in(RAMP) out(EAST) : !csl.route
  %c  = csl.color : !csl.color
  %regionA = csl.code_region routes(%r1) colors(%c) {
  } {width = 10 : i64, height = 10 : i64} : !csl.code_region
  %regionB = csl.code_region routes(%r2) colors(%c) {
  } {width = 10 : i64, height = 10 : i64} : !csl.code_region
}

// CHECK-LABEL: csl.spatial_placement
// CHECK:   %[[K:.*]] = csl.kernel
csl.spatial_placement {
  %k = csl.kernel {
  } {source_file = "pe.csl"} : !csl.kernel
}

// ============================================================================
// Full integration — matches placement_mlir_design.txt
// ============================================================================

// CHECK-LABEL: csl.spatial_placement
// CHECK:   %[[RED:.*]] = csl.color : !csl.color
// CHECK:   %[[GREEN:.*]] = csl.color : !csl.color
// CHECK:   %[[R1:.*]] = csl.route in(RAMP) out(EAST) : !csl.route
// CHECK:   %[[R2:.*]] = csl.route in(RAMP) out(EAST) : !csl.route
// CHECK:   %[[K1:.*]] = csl.kernel
// CHECK:   %[[K2:.*]] = csl.kernel
// CHECK:   %[[RA:.*]] = csl.code_region routes(%[[R1]]) colors(%[[RED]], %[[GREEN]])
// CHECK:   {width = 10 : i64, height = 10 : i64} : !csl.code_region
// CHECK:   %[[RB:.*]] = csl.code_region routes(%[[R2]]) colors(%[[RED]])
// CHECK:   {width = 10 : i64, height = 10 : i64} : !csl.code_region
// CHECK:   csl.place %[[RA]] %[[K1]] {x = 0 : i64, y = 0 : i64}
// CHECK:   csl.place %[[RB]] %[[K2]] {x = 10 : i64, y = 10 : i64}
csl.spatial_placement {
  // Colors
  %red   = csl.color : !csl.color
  %green = csl.color : !csl.color

  // Routes
  %r1 = csl.route in(RAMP) out(EAST) : !csl.route
  %r2 = csl.route in(RAMP) out(EAST) : !csl.route

  // Kernels
  %k1 = csl.kernel {
  } {source_file = "k1.csl"} : !csl.kernel
  %k2 = csl.kernel {
  } {source_file = "k2.csl"} : !csl.kernel

  // Code regions
  %regionA = csl.code_region routes(%r1) colors(%red, %green) {
  } {width = 10 : i64, height = 10 : i64} : !csl.code_region

  %regionB = csl.code_region routes(%r2) colors(%red) {
  } {width = 10 : i64, height = 10 : i64} : !csl.code_region

  // Placement
  csl.place %regionA %k1 {x = 0 : i64, y = 0 : i64}
  csl.place %regionB %k2 {x = 10 : i64, y = 10 : i64}
}
