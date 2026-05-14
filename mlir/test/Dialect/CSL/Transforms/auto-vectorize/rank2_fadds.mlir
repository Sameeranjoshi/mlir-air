// RUN: air-opt %s -csl-auto-vectorize | FileCheck %s
//
// 2D matrix add: C[i,j] = A[i,j] + B[i,j]  -> @fadds on mem4d_dsd (rank-2).

// CHECK-LABEL: csl.func @compute
// CHECK-NOT: scf.for
// All accesses have offset=[0,0] stride=[1,1] so buildSubviewForAccess
// returns the raw memref directly (no subview emitted).
// CHECK-NOT: memref.subview
// CHECK: %[[DA:.+]] = csl.get_mem_dsd %A
// CHECK: %[[DB:.+]] = csl.get_mem_dsd %B
// CHECK: %[[DC:.+]] = csl.get_mem_dsd %C
// CHECK: csl.builtin_call "fadds"(%[[DC]], %[[DA]], %[[DB]])

module {
  csl.wafer @mat2 {arch = "wse3"} {
    csl.program @pe {
      %A = csl.var @A : memref<8x16xf32>
      %B = csl.var @B : memref<8x16xf32>
      %C = csl.var @C : memref<8x16xf32>
      csl.func @compute {
        %c0 = arith.constant 0 : index
        %M  = arith.constant 8 : index
        %N  = arith.constant 16 : index
        %c1 = arith.constant 1 : index
        scf.for %i = %c0 to %M step %c1 {
          scf.for %j = %c0 to %N step %c1 {
            %va = memref.load %A[%i, %j] : memref<8x16xf32>
            %vb = memref.load %B[%i, %j] : memref<8x16xf32>
            %vc = arith.addf %va, %vb : f32
            memref.store %vc, %C[%i, %j] : memref<8x16xf32>
          }
        }
        csl.return
      }
    }
    csl.layout {width = 1 : i64, height = 1 : i64} @layout {
      csl_layout.place @pe at (0, 0)
    }
    csl.host @main() {layout = @layout} {
    }
  }
}
