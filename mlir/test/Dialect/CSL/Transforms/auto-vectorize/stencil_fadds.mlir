// RUN: air-opt %s -csl-auto-vectorize | FileCheck %s
//
// 2-point stencil: c[i] = a[i-1] + a[i]  for i in [1, N-1)
// The loop is clipped to keep a[i-1] in bounds.  The pass must emit two
// subviews of `a` (offsets 0 and 1) and one of `c` (offset 1).

// CHECK-LABEL: csl.func @compute
// CHECK-NOT: scf.for
// a[i-1]: offset=0 stride=1 → no subview (raw buffer passed directly)
// a[i]:   offset=1 stride=1 → subview %a[1][126][1]
// c[i]:   offset=1 stride=1 → subview %c[1][126][1]
// CHECK: memref.subview %a[1]
// CHECK: memref.subview %c[1]
// CHECK: csl.get_mem_dsd %a
// CHECK: csl.get_mem_dsd
// CHECK: csl.get_mem_dsd
// CHECK: csl.builtin_call "fadds"

module {
  csl.wafer @stencil2 {arch = "wse3"} {
    csl.program @pe {
      %a = csl.var @a : memref<128xf32>
      %c = csl.var @c : memref<128xf32>
      csl.func @compute {
        %c1 = arith.constant 1 : index
        %Nm1 = arith.constant 127 : index
        scf.for %i = %c1 to %Nm1 step %c1 {
          %im1 = arith.subi %i, %c1 : index
          %vl = memref.load %a[%im1] : memref<128xf32>
          %vc = memref.load %a[%i]   : memref<128xf32>
          %s  = arith.addf %vl, %vc  : f32
          memref.store %s, %c[%i]    : memref<128xf32>
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
