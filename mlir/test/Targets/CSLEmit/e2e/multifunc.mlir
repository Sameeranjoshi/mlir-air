// RUN: air-opt %s -csl-infer-exports | air-translate --emit-csl-program | FileCheck --check-prefix=HELPER %s
//
// Verifies that func.func private helpers emit as CSL fn declarations,
// func.call emits as variable assignments, func.return emits return stmts,
// and the <layout> library import is auto-injected.

// HELPER:      const layout_mod = @import_module("<layout>");
// HELPER:      fn scaled_add(a0: f32, a1: f32, a2: f32) f32 {
// HELPER-NEXT:   var t{{[0-9]+}}: f32 = a0 * a2;
// HELPER:        var t{{[0-9]+}}: f32 = t{{[0-9]+}} + a1;
// HELPER:        return t{{[0-9]+}};
// HELPER:      }
// HELPER:      fn compute() void {
// HELPER:        var t{{[0-9]+}}: f32 = scaled_add({{.*}}, {{.*}}, {{.*}});

module {
  csl.wafer @helper_add {arch = "wse3"} {
    csl.program @pe {
      %a = csl.var @a : memref<256xf32>
      %b = csl.var @b : memref<256xf32>
      %c = csl.var @c : memref<256xf32>

      // Private helper: computes x*s + y
      func.func private @scaled_add(%x: f32, %y: f32, %s: f32) -> f32 {
        %m = arith.mulf %x, %s : f32
        %r = arith.addf %m, %y : f32
        func.return %r : f32
      }

      csl.func @compute {
        %c0 = arith.constant 0 : index
        %n = arith.constant 256 : index
        %c1 = arith.constant 1 : index
        %scale = arith.constant 2.0 : f32
        scf.for %i = %c0 to %n step %c1 {
          %va = memref.load %a[%i] : memref<256xf32>
          %vb = memref.load %b[%i] : memref<256xf32>
          %vr = func.call @scaled_add(%va, %vb, %scale) : (f32, f32, f32) -> f32
          memref.store %vr, %c[%i] : memref<256xf32>
        }
        csl.return
      }

      csl.export @a {alias = "a"}
      csl.export @b {alias = "b"}
      csl.export @c {alias = "c"}
      csl.export @compute {kind = "func"}
    }
    csl.layout {width = 1 : i64, height = 1 : i64} @main_layout {
      csl_layout.place @pe at (0, 0)
    }
    csl.host @main(%a_in: memref<256xf32>, %b_in: memref<256xf32>,
                   %c_out: memref<256xf32>) {layout = @main_layout} {
      csl_host.memcpy_h2d %a_in to @main_layout::@a
          {px = 0 : i64, py = 0 : i64, width = 1 : i64, height = 1 : i64}
          : memref<256xf32>
      csl_host.memcpy_h2d %b_in to @main_layout::@b
          {px = 0 : i64, py = 0 : i64, width = 1 : i64, height = 1 : i64}
          : memref<256xf32>
      csl_host.launch @main_layout::@compute
      csl_host.memcpy_d2h @main_layout::@c to %c_out
          {px = 0 : i64, py = 0 : i64, width = 1 : i64, height = 1 : i64}
          : memref<256xf32>
    }
  }
}
