// RUN: split-file %s %t
// RUN: air-translate --emit-csl-layout %t/vecadd.mlir | FileCheck %s --check-prefix=CHECK-VECADD
// RUN: air-translate --emit-csl-layout %t/gemv.mlir   | FileCheck %s --check-prefix=CHECK-GEMV
//
// Verifies --emit-csl-layout output for a 1x1 vecadd program and a
// parameterized gemv program. As of 2026-04-20 this emits the cslc-input
// layout.csl wrapper (memcpy import + @set_rectangle + @set_tile_code +
// @export_name), not the dead csl_layout.py form.
//
// vecadd: checks @set_tile_code points at the program name, exports are
//         declared with the right element type + writable bit.
// gemv:   checks @set_tile_code threads the comptime param values.

// CHECK-VECADD: const memcpy = @import_module("<memcpy/get_params>", .{
// CHECK-VECADD:   .width = 1,
// CHECK-VECADD:   .height = 1,
// CHECK-VECADD: });
// CHECK-VECADD: layout {
// CHECK-VECADD:   @set_rectangle(1, 1);
// CHECK-VECADD:   @set_tile_code(0, 0, "vecadd_pe.csl", .{ .memcpy_params = memcpy.get_params(0) });
// CHECK-VECADD:   @export_name("a", [*]f32, true);
// CHECK-VECADD:   @export_name("b", [*]f32, true);
// CHECK-VECADD:   @export_name("c", [*]f32, true);
// CHECK-VECADD:   @export_name("compute", fn()void);
// CHECK-VECADD: }

// CHECK-GEMV: const memcpy = @import_module("<memcpy/get_params>", .{
// CHECK-GEMV:   .width = 4,
// CHECK-GEMV:   .height = 1,
// CHECK-GEMV: });
// CHECK-GEMV: layout {
// CHECK-GEMV:   @set_rectangle(4, 1);
// CHECK-GEMV:   @set_tile_code(0, 0, "gemv_pe.csl", .{ .memcpy_params = memcpy.get_params(0) });
// (Threading of literal-int params on point-form placements is a separate
// gap — only iv-name params on range-form placements are wired today.)

//--- vecadd.mlir
module {
  csl.wafer @vecadd {arch = "wse3"} {
    csl.program @vecadd_pe {
      %a = csl.var @a : memref<256xf32>
      %b = csl.var @b : memref<256xf32>
      %c = csl.var @c : memref<256xf32>
      csl.func @compute { csl.return }
      csl.export @a {alias = "a"}
      csl.export @b {alias = "b"}
      csl.export @c {alias = "c"}
      csl.export @compute {kind = "func"}
    }
    csl.layout {width = 1 : i64, height = 1 : i64} @main_layout {
      csl_layout.place @vecadd_pe at (0, 0)
      csl_layout.export "a" from @vecadd_pe::@a
      csl_layout.export "b" from @vecadd_pe::@b
      csl_layout.export "c" from @vecadd_pe::@c
      csl_layout.export "compute" from @vecadd_pe::@compute {kind = "func"}
    }
    csl.host @main(%a_in: memref<256xf32>, %b_in: memref<256xf32>,
                   %c_out: memref<256xf32>) {layout = @main_layout} {
    }
  }
}

//--- gemv.mlir
module {
  csl.wafer @gemv {arch = "wse3"} {
    csl.program @gemv_pe(%M: !csl.comptime<i16>, %N: !csl.comptime<i16>) {
    }
    csl.layout {width = 4 : i64, height = 1 : i64} @gemv_layout {
      csl_layout.place @gemv_pe at (0, 0) {M = 4 : i16, N = 6 : i16}
    }
    csl.host @main() {layout = @gemv_layout} {
    }
  }
}
