//===- routing_ops.mlir - CSL routing dialect ops tests --------*- MLIR -*-===//
//
// Part of the air-to-csl project.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s | FileCheck %s

// CHECK-LABEL: module {

// Color declaration (no ID — compiler assigns)
// CHECK: %[[RED:.*]] = csl.color : !csl.color
// CHECK: %[[GREEN:.*]] = csl.color : !csl.color
%red   = csl.color : !csl.color
%green = csl.color : !csl.color

// Color with explicit ID
// CHECK: %[[C0:.*]] = csl.color 0 : !csl.color
// CHECK: %[[C1:.*]] = csl.color 1 : !csl.color
%c0 = csl.color 0 : !csl.color
%c1 = csl.color 1 : !csl.color

// Route with in/out directions
// CHECK: %[[R1:.*]] = csl.route in(RAMP) out(WEST) : i32
// CHECK: %[[R2:.*]] = csl.route in(RAMP) out(EAST) : i32
// CHECK: %[[R3:.*]] = csl.route in(RAMP) out(NORTH) : i32
// CHECK: %[[R4:.*]] = csl.route in(RAMP) out(SOUTH) : i32
%r1 = csl.route in(RAMP) out(WEST) : i32
%r2 = csl.route in(RAMP) out(EAST) : i32
%r3 = csl.route in(RAMP) out(NORTH) : i32
%r4 = csl.route in(RAMP) out(SOUTH) : i32

// Receiver routes (fabric -> PE)
// CHECK: %[[R5:.*]] = csl.route in(WEST) out(RAMP) : i32
// CHECK: %[[R6:.*]] = csl.route in(EAST) out(RAMP) : i32
%r5 = csl.route in(WEST) out(RAMP) : i32
%r6 = csl.route in(EAST) out(RAMP) : i32

// Pass-through routes (fabric -> fabric)
// CHECK: %[[R7:.*]] = csl.route in(WEST) out(EAST) : i32
%r7 = csl.route in(WEST) out(EAST) : i32
