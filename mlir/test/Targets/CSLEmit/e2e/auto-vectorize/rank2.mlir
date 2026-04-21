// RUN: rm -rf %t && mkdir -p %t
// RUN: air-opt %s -csl-auto-vectorize -csl-infer-exports | \
// RUN:   air-translate --emit-csl --output-dir=%t 2>&1 | FileCheck %s --check-prefix=EMIT-ERR
// XFAIL: *
//
// XFAIL: The emitter does not yet support 2-D csl.var — it rejects the op
// with "csl.var must have 1-D memref type".  Once the emitter handles
// multi-dimensional vars (flattened to a 1-D Zig array), this test should
// be promoted to a CHECK-LABEL on pe.csl for @get_dsd(mem4d_dsd, ...) and
// @fadds(.
//
// EMIT-ERR: csl.var must have 1-D memref type

module {
  csl.wafer @mat2_auto {arch = "wse3"} {
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
    csl.host @main(%Ai: memref<8x16xf32>, %Bi: memref<8x16xf32>,
                   %Co: memref<8x16xf32>) {layout = @layout} {
      csl_host.memcpy_h2d %Ai to @layout::@A
          {px = 0 : i64, py = 0 : i64, width = 1 : i64, height = 1 : i64}
          : memref<8x16xf32>
      csl_host.memcpy_h2d %Bi to @layout::@B
          {px = 0 : i64, py = 0 : i64, width = 1 : i64, height = 1 : i64}
          : memref<8x16xf32>
      csl_host.launch @layout::@compute
      csl_host.memcpy_d2h @layout::@C to %Co
          {px = 0 : i64, py = 0 : i64, width = 1 : i64, height = 1 : i64}
          : memref<8x16xf32>
    }
  }
}
