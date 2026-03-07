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
// CHECK:   %[[R1:.*]] = csl.route in(RAMP) out(WEST) : i32
// CHECK:   %[[RED:.*]] = csl.color : !csl.color
// CHECK:   %[[REG:.*]] = csl.code_region routes(%[[R1]]) colors(%[[RED]]) shape(10, 10)
// CHECK:   } : !csl.code_region
csl.spatial_placement {
  %r1  = csl.route in(RAMP) out(WEST) : i32
  %red = csl.color : !csl.color
  %regionA = csl.code_region routes(%r1) colors(%red) shape(10, 10) {
  } : !csl.code_region
}

// ============================================================================
// csl.paint — paint a PE inside a code_region body
// ============================================================================

// CHECK-LABEL: csl.spatial_placement
// CHECK:   csl.paint pe(0, 0) route(%{{.*}}) color(%{{.*}})
csl.spatial_placement {
  %r1  = csl.route in(RAMP) out(EAST) : i32
  %red = csl.color : !csl.color
  %regionA = csl.code_region routes(%r1) colors(%red) shape(4, 4) {
    csl.paint pe(0, 0) route(%r1) color(%red)
  } : !csl.code_region
}

// ============================================================================
// csl.paint — multiple paint ops inside body
// ============================================================================

// CHECK-LABEL: csl.spatial_placement
// CHECK:   csl.paint pe(0, 0) route(%{{.*}}) color(%{{.*}})
// CHECK:   csl.paint pe(1, 0) route(%{{.*}}) color(%{{.*}})
csl.spatial_placement {
  %r1    = csl.route in(RAMP) out(EAST) : i32
  %r2    = csl.route in(RAMP) out(WEST) : i32
  %red   = csl.color : !csl.color
  %green = csl.color : !csl.color
  %regionA = csl.code_region routes(%r1, %r2) colors(%red, %green) shape(2, 1) {
    csl.paint pe(0, 0) route(%r1) color(%red)
    csl.paint pe(1, 0) route(%r2) color(%green)
  } : !csl.code_region
}

// ============================================================================
// csl.port — declare ports on a code region
// ============================================================================

// CHECK-LABEL: csl.spatial_placement
// CHECK:   %[[REG:.*]] = csl.code_region
// CHECK:   %[[P1:.*]] = csl.port %[[REG]] type("input") route(%{{.*}}) size(10) : !csl.port
// CHECK:   %[[P2:.*]] = csl.port %[[REG]] type("output") route(%{{.*}}) size(10) : !csl.port
csl.spatial_placement {
  %r1  = csl.route in(RAMP) out(WEST) : i32
  %r2  = csl.route in(RAMP) out(EAST) : i32
  %red = csl.color : !csl.color
  %regionA = csl.code_region routes(%r1, %r2) colors(%red) shape(10, 10) {
    csl.paint pe(0, 0) route(%r1) color(%red)
  } : !csl.code_region
  %port1 = csl.port %regionA type("input")  route(%r1) size(10) : !csl.port
  %port2 = csl.port %regionA type("output") route(%r2) size(10) : !csl.port
}

// ============================================================================
// csl.place — place a region at a coordinate with kernel
// ============================================================================

// CHECK-LABEL: csl.spatial_placement
// CHECK:   %[[REG:.*]] = csl.code_region
// CHECK:   %[[K:.*]] = csl.kernel "pe.csl"
// CHECK:   csl.place %[[REG]] at(0, 0) kernel(%[[K]])
csl.spatial_placement {
  %r1  = csl.route in(RAMP) out(EAST) : i32
  %red = csl.color : !csl.color
  %regionA = csl.code_region routes(%r1) colors(%red) shape(10, 10) {
  } : !csl.code_region
  %k1 = csl.kernel "pe.csl" {
    csl.func @compute() : () -> () { csl.return }
  } : !csl.kernel
  csl.place %regionA at(0, 0) kernel(%k1)
}

// ============================================================================
// csl.dataflow — connect ports across regions
// ============================================================================

// CHECK-LABEL: csl.spatial_placement
// CHECK:   csl.dataflow %{{.*}} -> %{{.*}}
csl.spatial_placement {
  %r1 = csl.route in(RAMP) out(EAST) : i32
  %r2 = csl.route in(WEST) out(RAMP) : i32
  %c  = csl.color : !csl.color
  %regionA = csl.code_region routes(%r1) colors(%c) shape(10, 10) {
  } : !csl.code_region
  %regionB = csl.code_region routes(%r2) colors(%c) shape(10, 10) {
  } : !csl.code_region
  %tx = csl.port %regionA type("output") route(%r1) size(10) : !csl.port
  %rx = csl.port %regionB type("input")  route(%r2) size(10) : !csl.port
  csl.dataflow %tx -> %rx
}

// ============================================================================
// csl.kernel — kernel with params
// ============================================================================

// CHECK-LABEL: csl.spatial_placement
// CHECK:   %[[K:.*]] = csl.kernel "pe.csl" params({param1 = 100 : i32})
csl.spatial_placement {
  %k = csl.kernel "pe.csl" params({param1 = 100 : i32}) {
    csl.var @buf : memref<1024xf32>
    csl.func @compute() : () -> () { csl.return }
  } : !csl.kernel
}

// ============================================================================
// Full integration — matches placement_mlir_design.txt
// ============================================================================

// CHECK-LABEL: csl.spatial_placement
// CHECK:   %[[RED:.*]] = csl.color : !csl.color
// CHECK:   %[[GREEN:.*]] = csl.color : !csl.color
// CHECK:   %[[R1:.*]] = csl.route in(RAMP) out(EAST) : i32
// CHECK:   %[[R2:.*]] = csl.route in(WEST) out(RAMP) : i32
// CHECK:   %[[K1:.*]] = csl.kernel "k1.csl"
// CHECK:   %[[K2:.*]] = csl.kernel "k2.csl"
// CHECK:   %[[RA:.*]] = csl.code_region routes(%[[R1]]) colors(%[[RED]], %[[GREEN]]) shape(10, 10)
// CHECK:     csl.paint pe(0, 0) route(%[[R1]]) color(%[[RED]])
// CHECK:   } : !csl.code_region
// CHECK:   %[[RB:.*]] = csl.code_region routes(%[[R2]]) colors(%[[RED]]) shape(10, 10)
// CHECK:   } : !csl.code_region
// CHECK:   %[[P1:.*]] = csl.port %[[RA]] type("output") route(%[[R1]]) size(256) : !csl.port
// CHECK:   %[[P2:.*]] = csl.port %[[RB]] type("input") route(%[[R2]]) size(256) : !csl.port
// CHECK:   csl.place %[[RA]] at(0, 0) kernel(%[[K1]])
// CHECK:   csl.place %[[RB]] at(10, 10) kernel(%[[K2]])
// CHECK:   csl.dataflow %[[P1]] -> %[[P2]]
csl.spatial_placement {
  // Colors
  %red   = csl.color : !csl.color
  %green = csl.color : !csl.color

  // Routes
  %r1 = csl.route in(RAMP) out(EAST) : i32
  %r2 = csl.route in(WEST) out(RAMP) : i32

  // Kernels
  %k1 = csl.kernel "k1.csl" params({param1 = 100 : i32}) {
    csl.func @compute() : () -> () { csl.return }
  } : !csl.kernel
  %k2 = csl.kernel "k2.csl" params({param2 = 200 : i32}) {
    csl.func @compute() : () -> () { csl.return }
  } : !csl.kernel

  // Code regions with paint
  %regionA = csl.code_region routes(%r1) colors(%red, %green) shape(10, 10) {
    csl.paint pe(0, 0) route(%r1) color(%red)
  } : !csl.code_region

  %regionB = csl.code_region routes(%r2) colors(%red) shape(10, 10) {
    csl.paint pe(0, 0) route(%r2) color(%red)
  } : !csl.code_region

  // Ports
  %port1 = csl.port %regionA type("output") route(%r1) size(256) : !csl.port
  %port2 = csl.port %regionB type("input")  route(%r2) size(256) : !csl.port

  // Placement
  csl.place %regionA at(0, 0)   kernel(%k1)
  csl.place %regionB at(10, 10) kernel(%k2)

  // Dataflow across regions
  csl.dataflow %port1 -> %port2
}
