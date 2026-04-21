// RUN: rm -rf %t && mkdir -p %t
// RUN: air-opt %s -csl-auto-vectorize -csl-infer-exports | \
// RUN:   air-translate --emit-csl --output-dir=%t
// RUN: FileCheck %s < %t/saxpy_auto/pe.csl
//
// Mirror dsds.mlir but with a scalar scf.for input: -csl-auto-vectorize
// detects the saxpy FMA idiom (A[i]*alpha + y[i] -> y[i]) and replaces it
// with @fmacs.
//
// CHECK-LABEL: fn compute() void
// CHECK: @fmacs(

module {
  csl.wafer @saxpy_auto {arch = "wse3"} {
    csl.program @pe {
      %A = csl.var @A : memref<128xf32>
      %y = csl.var @y : memref<128xf32>
      csl.func @compute {
        %alpha = arith.constant 2.0 : f32
        %c0 = arith.constant 0 : index
        %n  = arith.constant 128 : index
        %c1 = arith.constant 1 : index
        scf.for %i = %c0 to %n step %c1 {
          %va = memref.load %A[%i] : memref<128xf32>
          %vy = memref.load %y[%i] : memref<128xf32>
          %m  = arith.mulf %va, %alpha : f32
          %s  = arith.addf %m, %vy : f32
          memref.store %s, %y[%i] : memref<128xf32>
        }
        csl.return
      }
    }
    csl.layout {width = 1 : i64, height = 1 : i64} @layout {
      csl_layout.place @pe at (0, 0)
    }
    csl.host @main(%A_in: memref<128xf32>, %y_io: memref<128xf32>)
        {layout = @layout} {
      csl_host.memcpy_h2d %A_in to @layout::@A
          {px = 0 : i64, py = 0 : i64, width = 1 : i64, height = 1 : i64}
          : memref<128xf32>
      csl_host.memcpy_h2d %y_io to @layout::@y
          {px = 0 : i64, py = 0 : i64, width = 1 : i64, height = 1 : i64}
          : memref<128xf32>
      csl_host.launch @layout::@compute
      csl_host.memcpy_d2h @layout::@y to %y_io
          {px = 0 : i64, py = 0 : i64, width = 1 : i64, height = 1 : i64}
          : memref<128xf32>
    }
  }
}
