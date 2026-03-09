//===- roundtrip.mlir - CSL dialect round-trip tests -----------*- MLIR -*-===//
//
// Part of the air-to-csl project.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
//
// Verifies that the CSL dialect ops survive a parse-print-parse round trip
// using --verify-roundtrip.
//
//===----------------------------------------------------------------------===//

// RUN: air-opt --verify-roundtrip %s | FileCheck %s

// ---- Spatial placement with routing and kernels ----

// CHECK-LABEL: csl.spatial_placement
// CHECK:   %[[C:.*]] = csl.color : !csl.color
// CHECK:   %[[R:.*]] = csl.route in(RAMP) out(EAST) : !csl.route
// CHECK:   %[[K:.*]] = csl.kernel
// CHECK:   %[[REG:.*]] = csl.code_region routes(%[[R]]) colors(%[[C]])
// CHECK:   } : !csl.code_region
// CHECK:   csl.place %[[REG]] %[[K]] {x = 0 : i64, y = 0 : i64}
csl.spatial_placement {
  %c = csl.color : !csl.color
  %r = csl.route in(RAMP) out(EAST) : !csl.route
  %k = csl.kernel {
  } {source_file = "sender.csl"} : !csl.kernel
  %reg = csl.code_region routes(%r) colors(%c) {
  } {width = 4 : i64, height = 4 : i64} : !csl.code_region
  csl.place %reg %k {x = 0 : i64, y = 0 : i64}
}


// ---- Color with explicit ID ----

// CHECK: %[[C0:.*]] = csl.color {id = 0 : i32} : !csl.color
%c0 = csl.color {id = 0 : i32} : !csl.color

// ---- Route directions ----

// CHECK: %[[R1:.*]] = csl.route in(RAMP) out(EAST) : !csl.route
// CHECK: %[[R2:.*]] = csl.route in(RAMP) out(EAST) : !csl.route
%r1 = csl.route in(RAMP) out(EAST) : !csl.route
%r2 = csl.route in(RAMP) out(EAST) : !csl.route

// Data movement ops are deferred — commented out in ODS (csl.get_mem_dsd, csl.get_fab_dsd, csl.mov)
