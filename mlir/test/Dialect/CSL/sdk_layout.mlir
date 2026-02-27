//===- sdk_layout.mlir - SdkLayout dialect ops round-trip test --*- MLIR -*-===//
//
// Part of the air-to-csl project.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
//
// SdkLayout-specific ops are currently disabled (guarded behind
// CSL_ENABLE_SDK_LAYOUT_OPS). This test is a placeholder.
//
// When SdkLayout ops are re-enabled, restore the full round-trip tests.
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s | FileCheck %s

// CHECK-LABEL: module
module {}
