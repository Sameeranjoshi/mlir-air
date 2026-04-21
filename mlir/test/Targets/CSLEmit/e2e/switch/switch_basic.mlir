// RUN: rm -rf %t && mkdir -p %t
// RUN: air-opt %s -csl-infer-exports | \
// RUN:   air-translate --emit-csl --output-dir=%t
// RUN: FileCheck %s < %t/switch_basic_e2e/pe.csl
//
// Emit scf.index_switch as a CSL switch statement.
// Each element c[i] is computed by branching on i % 3:
//   case 0: a[i] + b[i]
//   case 1: a[i] - b[i]
//   else:   a[i] * b[i]

// CHECK-LABEL: fn compute() void
// CHECK: switch (
// CHECK: 0 => {
// CHECK: 1 => {
// CHECK: else => {

module {
  csl.wafer @switch_basic_e2e {arch = "wse3"} {
    csl.program @pe {
      %a = csl.var @a : memref<64xf32>
      %b = csl.var @b : memref<64xf32>
      %c = csl.var @c : memref<64xf32>
      csl.func @compute {
        %c0 = arith.constant 0 : index
        %n  = arith.constant 64 : index
        %c1 = arith.constant 1 : index
        %c3 = arith.constant 3 : index
        scf.for %i = %c0 to %n step %c1 {
          %va = memref.load %a[%i] : memref<64xf32>
          %vb = memref.load %b[%i] : memref<64xf32>
          %mod = arith.remsi %i, %c3 : index
          scf.index_switch %mod
          case 0 {
            %s = arith.addf %va, %vb : f32
            memref.store %s, %c[%i] : memref<64xf32>
            scf.yield
          }
          case 1 {
            %s = arith.subf %va, %vb : f32
            memref.store %s, %c[%i] : memref<64xf32>
            scf.yield
          }
          default {
            %s = arith.mulf %va, %vb : f32
            memref.store %s, %c[%i] : memref<64xf32>
          }
        }
        csl.return
      }
    }
    csl.layout {width = 1 : i64, height = 1 : i64} @layout {
      csl_layout.place @pe at (0, 0)
    }
    csl.host @main(%a_in: memref<64xf32>, %b_in: memref<64xf32>, %c_out: memref<64xf32>)
        {layout = @layout} {
      csl_host.memcpy_h2d %a_in to @layout::@a
          {px = 0 : i64, py = 0 : i64, width = 1 : i64, height = 1 : i64}
          : memref<64xf32>
      csl_host.memcpy_h2d %b_in to @layout::@b
          {px = 0 : i64, py = 0 : i64, width = 1 : i64, height = 1 : i64}
          : memref<64xf32>
      csl_host.launch @layout::@compute
      csl_host.memcpy_d2h @layout::@c to %c_out
          {px = 0 : i64, py = 0 : i64, width = 1 : i64, height = 1 : i64}
          : memref<64xf32>
    }
  }
}
