// Test: two herds with async dependency (matmul -> add).
// - Herd sizes must be arith.constant (not segment args); dependency canonicalizer
//   calls getNumRows()/getNumCols() which cast size operands to ConstantIndexOp.
// - No scf.for inside herds to avoid printer crash from block-arg mismatch.
//
// RUN: air-opt %s -air-dependency -canonicalize -air-dependency-canonicalize -air-dependency-parse-graph | FileCheck %s
// CHECK: air.herd @mm
// CHECK: air.herd @add async

module {
  func.func @matmul_add(%A: memref<64x64xf32>, %B: memref<64x64xf32>,
                        %C: memref<64x64xf32>, %D: memref<64x64xf32>) {
    air.segment args(%A_in = %A, %B_in = %B, %C_in = %C, %D_in = %D) : memref<64x64xf32>, memref<64x64xf32>, memref<64x64xf32>, memref<64x64xf32> {
      %c4_seg = arith.constant 4 : index
      // Herd 0: one tile of matmul (single element at 0,0 for minimal body)
      %t0 = air.herd @mm async tile(%tx, %ty) in (%sx = %c4_seg, %sy = %c4_seg) args(%a = %A_in, %b = %B_in, %c = %C_in) : memref<64x64xf32>, memref<64x64xf32>, memref<64x64xf32> {
        %c0 = arith.constant 0 : index
        %val_a = memref.load %a[%c0, %c0] : memref<64x64xf32>
        %val_b = memref.load %b[%c0, %c0] : memref<64x64xf32>
        %prod = arith.mulf %val_a, %val_b : f32
        memref.store %prod, %c[%c0, %c0] : memref<64x64xf32>
        air.herd_terminator
      }

      // Herd 1: add 1.0, depends on herd 0
      %t1 = air.herd @add async [%t0] tile(%tx, %ty) in (%sx = %c4_seg, %sy = %c4_seg) args(%c = %C_in, %d = %D_in) : memref<64x64xf32>, memref<64x64xf32> {
        %c0 = arith.constant 0 : index
        %val_c = memref.load %c[%c0, %c0] : memref<64x64xf32>
        %one = arith.constant 1.0 : f32
        %result = arith.addf %val_c, %one : f32
        memref.store %result, %d[%c0, %c0] : memref<64x64xf32>
        air.herd_terminator
    }

      air.segment_terminator
    }
    return
  }
}
