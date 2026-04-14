//===- vecadd_layout_emit.mlir - LayoutEmitter FileCheck test ---*- MLIR -*-===//
//
// Part of the air-to-csl project.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
//
// Verifies that --emit-csl-rt --csl-output-dir writes a correct layout.csl.
//
// RUN: air-translate --emit-csl-rt --csl-output-dir=%t %s -o /dev/null
// RUN: cat %t/layout.csl | FileCheck %s
//
// CHECK: const memcpy = @import_module("<memcpy/get_params>", .{ .width = 1, .height = 1 });
// CHECK: layout {
// CHECK:   @set_rectangle(1, 1);
// CHECK:   @set_tile_code(0, 0, "vecadd_pe.csl"
// CHECK-DAG: @export_name("a", [*]f32, true);
// CHECK-DAG: @export_name("b", [*]f32, true);
// CHECK-DAG: @export_name("c", [*]f32, false);
// CHECK-DAG: @export_name("compute", fn()void);
// CHECK: }
//
//===----------------------------------------------------------------------===//

module {
  func.func @vecadd(%a: memref<256xf32>, %b: memref<256xf32>, %c: memref<256xf32>) {
    csl.spatial_placement {
      %k = csl.kernel {
        %a_buf = csl.var @a_buf : memref<256xf32>
        %b_buf = csl.var @b_buf : memref<256xf32>
        %c_buf = csl.var @c_buf : memref<256xf32>
      } {source_file = "vecadd_pe.csl"} : !csl.kernel
      %r = csl.code_region routes() colors() {
      }{width = 1 : i64, height = 1 : i64} : !csl.code_region
      csl.place %r %k {x = 0 : i64, y = 0 : i64}
    }
    csl.export_name "a" : memref<256xf32>{direction = "in"}
    csl.export_name "b" : memref<256xf32>{direction = "in"}
    csl.export_name "c" : memref<256xf32>{direction = "out"}
    csl.export_name "compute" : () -> ()
    return
  }
}
