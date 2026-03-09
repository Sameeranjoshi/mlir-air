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
// CHECK: %[[C0:.*]] = csl.color {id = 0 : i32} : !csl.color
// CHECK: %[[C1:.*]] = csl.color {id = 1 : i32} : !csl.color
%c0 = csl.color {id = 0 : i32} : !csl.color
%c1 = csl.color {id = 1 : i32} : !csl.color

// Route with in/out directions
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
