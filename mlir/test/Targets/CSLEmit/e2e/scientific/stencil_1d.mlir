// RUN: rm -rf %t && mkdir -p %t
// RUN: air-opt %s -csl-infer-exports | air-translate --emit-csl --output-dir=%t
// RUN: FileCheck %s < %t/stencil_1d/pe.csl
//
// v5 Task 7 — 1-D 3-point stencil WITHOUT halo (single PE).
//
//   y[0]     = x[0]
//   y[N-1]   = x[N-1]
//   y[i]     = 0.5 * x[i-1] + x[i] + 0.5 * x[i+1]    for 0 < i < N-1
//
// Exercises scf.for + scf.if (endpoint check) + arith.cmpi + index ops.

// CHECK-LABEL: fn compute() void
// CHECK: while (
// CHECK: if (
// CHECK: else
// CHECK: y[

module {
  csl.wafer @stencil_1d {arch = "wse3"} {
    csl.program @pe {
      %x = csl.var @x : memref<128xf32>
      %y = csl.var @y : memref<128xf32>
      csl.func @compute {
        %c0   = arith.constant 0   : index
        %c1   = arith.constant 1   : index
        %cN   = arith.constant 128 : index
        %cNm1 = arith.constant 127 : index
        %half = arith.constant 0.5 : f32
        scf.for %i = %c0 to %cN step %c1 {
          // Endpoint check: i == 0 || i == N-1.
          %ii0  = arith.cmpi eq, %i, %c0   : index
          %iiN  = arith.cmpi eq, %i, %cNm1 : index
          %is_b = arith.ori %ii0, %iiN     : i1
          scf.if %is_b {
            %v = memref.load %x[%i] : memref<128xf32>
            memref.store %v, %y[%i] : memref<128xf32>
          } else {
            %im = arith.subi %i, %c1 : index
            %ip = arith.addi %i, %c1 : index
            %vm = memref.load %x[%im] : memref<128xf32>
            %vi = memref.load %x[%i]  : memref<128xf32>
            %vp = memref.load %x[%ip] : memref<128xf32>
            %l  = arith.mulf %half, %vm : f32
            %r  = arith.mulf %half, %vp : f32
            %s1 = arith.addf %l, %vi    : f32
            %s2 = arith.addf %s1, %r    : f32
            memref.store %s2, %y[%i] : memref<128xf32>
          }
        }
        csl.return
      }
      csl.export @x {alias = "x"}
      csl.export @y {alias = "y"}
      csl.export @compute {kind = "func"}
    }
    csl.layout {width = 1 : i64, height = 1 : i64} @layout {
      csl_layout.place @pe at (0, 0)
    }
    csl.host @main(%x_in: memref<128xf32>, %y_out: memref<128xf32>)
        {layout = @layout} {
      csl_host.memcpy_h2d %x_in to @layout::@x
          {px = 0 : i64, py = 0 : i64, width = 1 : i64, height = 1 : i64}
          : memref<128xf32>
      csl_host.launch @layout::@compute
      csl_host.memcpy_d2h @layout::@y to %y_out
          {px = 0 : i64, py = 0 : i64, width = 1 : i64, height = 1 : i64}
          : memref<128xf32>
    }
  }
}
