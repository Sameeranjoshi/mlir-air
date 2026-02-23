//===- routing_ops.mlir - CSL routing dialect ops tests --------*- MLIR -*-===//
//
// Part of the air-to-csl project.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s | FileCheck %s

// CHECK-LABEL: module {

// Color declarations
// CHECK: %[[C0:.*]] = csl.color 0 : !csl.color
// CHECK: %[[C1:.*]] = csl.color 1 : !csl.color
%c0 = csl.color 0 : !csl.color
%c1 = csl.color 1 : !csl.color

// Routing in all five directions
// CHECK: csl.route %[[C0]] dir(EAST)
// CHECK: csl.route %[[C1]] dir(SOUTH)
csl.route %c0 dir(EAST)
csl.route %c1 dir(SOUTH)

// CHECK: %[[C2:.*]] = csl.color 2 : !csl.color
%c2 = csl.color 2 : !csl.color

// CHECK: csl.route %[[C2]] dir(NORTH)
// CHECK: csl.route %[[C2]] dir(WEST)
// CHECK: csl.route %[[C2]] dir(RAMP)
csl.route %c2 dir(NORTH)
csl.route %c2 dir(WEST)
csl.route %c2 dir(RAMP)

// Multiple colors with same direction
// CHECK: %[[C3:.*]] = csl.color 3 : !csl.color
// CHECK: %[[C4:.*]] = csl.color 4 : !csl.color
// CHECK: csl.route %[[C3]] dir(EAST)
// CHECK: csl.route %[[C4]] dir(EAST)
%c3 = csl.color 3 : !csl.color
%c4 = csl.color 4 : !csl.color
csl.route %c3 dir(EAST)
csl.route %c4 dir(EAST)

// High color IDs (WSE supports many colors)
// CHECK: %[[C24:.*]] = csl.color 24 : !csl.color
// CHECK: csl.route %[[C24]] dir(RAMP)
%c24 = csl.color 24 : !csl.color
csl.route %c24 dir(RAMP)
