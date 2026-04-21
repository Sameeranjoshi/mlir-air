// RUN: air-opt %s -csl-auto-vectorize | FileCheck %s
//
// c[i] = a[i] + b[i]  →  @fadds(dc, da, db)

// CHECK-LABEL: csl.func @compute
// CHECK-NOT: scf.for
// CHECK: %[[DA:.+]] = csl.get_mem_dsd %a : memref<1024xf32> -> !csl.dsd
// CHECK: %[[DB:.+]] = csl.get_mem_dsd %b : memref<1024xf32> -> !csl.dsd
// CHECK: %[[DC:.+]] = csl.get_mem_dsd %c : memref<1024xf32> -> !csl.dsd
// CHECK: csl.builtin_call "fadds"(%[[DC]], %[[DA]], %[[DB]]) : (!csl.dsd, !csl.dsd, !csl.dsd) -> ()

module {
  csl.wafer @vecadd {arch = "wse3"} {
    csl.program @pe {
      %a = csl.var @a : memref<1024xf32>
      %b = csl.var @b : memref<1024xf32>
      %c = csl.var @c : memref<1024xf32>
      csl.func @compute {
        %c0 = arith.constant 0 : index
        %n  = arith.constant 1024 : index
        %c1 = arith.constant 1 : index
        scf.for %i = %c0 to %n step %c1 {
          %va = memref.load %a[%i] : memref<1024xf32>
          %vb = memref.load %b[%i] : memref<1024xf32>
          %vc = arith.addf %va, %vb : f32
          memref.store %vc, %c[%i] : memref<1024xf32>
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
