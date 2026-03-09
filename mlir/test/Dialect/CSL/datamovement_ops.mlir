//===- datamovement_ops.mlir - CSL data movement ops tests ----*- MLIR -*-===//
//
// Part of the air-to-csl project.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
//
// DEFERRED: Data movement ops (csl.get_mem_dsd, csl.get_fab_dsd, csl.mov) are
// not yet implemented. These tests are disabled pending DSD/DSL implementation.
// See MEMORY.md for deferral rationale.
//
//===----------------------------------------------------------------------===//

// NOTE: This file is for documentation only. Tests are disabled because
// the data movement ops are deferred (commented out in CSLOps.td).
// To re-enable: uncomment ops in CSLOps.td and add RUN directive below.
//
// RUN: echo "DEFERRED - data movement ops not yet implemented"
//
// Reference implementations (commented out for deferred implementation):
//   csl.get_mem_dsd - Create DSD for memory
//   csl.get_fab_dsd - Create DSD for fabric (input/output)
//   csl.mov - Data movement operation between DSDs
