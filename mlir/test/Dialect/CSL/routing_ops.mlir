//===- routing_ops.mlir - CSL routing dialect ops tests --------*- MLIR -*-===//
//
// Part of the air-to-csl project.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s | FileCheck %s

// CHECK-LABEL: csl.wafer @routing_test
csl.wafer @routing_test {arch = "wse3"} {
  csl.program @p { csl.func @c { csl.return } }

  // Color declarations now live inside csl.layout and have a symbol name.
  // CHECK: csl.layout
  csl.layout {width = 1 : i64, height = 1 : i64} @layout {
    // Color without ID — compiler assigns
    // CHECK: csl.color @red
    // CHECK: csl.color @green
    csl.color @red
    csl.color @green

    // Color with explicit ID
    // CHECK: csl.color @c0 {id = 0 : i32}
    // CHECK: csl.color @c1 {id = 1 : i32}
    csl.color @c0 {id = 0 : i32}
    csl.color @c1 {id = 1 : i32}

    csl_layout.place @p at (0, 0)
  }
}

// Routes are still top-level SSA-valued ops (untouched by Task 1).
// CHECK: %[[R1:.*]] = csl.route in(RAMP) out(EAST) : !csl.route
// CHECK: %[[R2:.*]] = csl.route in(RAMP) out(EAST) : !csl.route
// CHECK: %[[R3:.*]] = csl.route in(RAMP) out(EAST) : !csl.route
// CHECK: %[[R4:.*]] = csl.route in(RAMP) out(EAST) : !csl.route
%r1 = csl.route in(RAMP) out(EAST) : !csl.route
%r2 = csl.route in(RAMP) out(EAST) : !csl.route
%r3 = csl.route in(RAMP) out(EAST) : !csl.route
%r4 = csl.route in(RAMP) out(EAST) : !csl.route

// Receiver routes (fabric -> PE)
// CHECK: %[[R5:.*]] = csl.route in(RAMP) out(EAST) : !csl.route
// CHECK: %[[R6:.*]] = csl.route in(RAMP) out(EAST) : !csl.route
%r5 = csl.route in(RAMP) out(EAST) : !csl.route
%r6 = csl.route in(RAMP) out(EAST) : !csl.route

// Pass-through routes (fabric -> fabric)
// CHECK: %[[R7:.*]] = csl.route in(RAMP) out(EAST) : !csl.route
%r7 = csl.route in(RAMP) out(EAST) : !csl.route
