//===- vecadd_kernel_emit.mlir - KernelEmitter FileCheck test ---*- MLIR -*-===//
//
// Part of the air-to-csl project.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
//
// Verifies that --emit-csl-rt --csl-output-dir writes a correct vecadd_pe.csl.
//
// RUN: air-opt %s | air-translate --emit-csl-rt --csl-output-dir=%t -o /dev/null
// RUN: cat %t/vecadd_pe.csl | FileCheck %s
//
// CHECK: param memcpy_params: comptime_struct
// CHECK: const sys_mod = @import_module("<memcpy/memcpy>", memcpy_params)
// CHECK: const N: i32 = 256
// CHECK-DAG: var a_buf: [N]f32
// CHECK-DAG: var b_buf: [N]f32
// CHECK-DAG: var c_buf: [N]f32
// CHECK-DAG: var a_ptr: [*]f32 = &a_buf
// CHECK-DAG: var b_ptr: [*]f32 = &b_buf
// CHECK-DAG: const c_ptr: [*]f32 = &c_buf
// CHECK: fn compute() void {
// CHECK:   for (@range(i32, 256)) |{{.*}}| {
// CHECK:     c_buf[{{.*}}] = a_buf[{{.*}}] + b_buf[{{.*}}];
// CHECK:   }
// CHECK:   sys_mod.unblock_cmd_stream();
// CHECK: }
// CHECK: comptime {
// CHECK-DAG: @export_symbol(a_ptr, "a");
// CHECK-DAG: @export_symbol(b_ptr, "b");
// CHECK-DAG: @export_symbol(c_ptr, "c");
// CHECK-DAG: @export_symbol(compute);
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
        csl.func @compute {
          %lo = arith.constant 0 : index
          %hi = arith.constant 256 : index
          %step = arith.constant 1 : index
          scf.for %i = %lo to %hi step %step {
            %va = memref.load %a_buf[%i] : memref<256xf32>
            %vb = memref.load %b_buf[%i] : memref<256xf32>
            %vc = arith.addf %va, %vb : f32
            memref.store %vc, %c_buf[%i] : memref<256xf32>
          }
          csl.return
        }
        csl.export_symbol @a_buf alias("a")
        csl.export_symbol @b_buf alias("b")
        csl.export_symbol @c_buf alias("c")
        csl.export_symbol @compute
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
