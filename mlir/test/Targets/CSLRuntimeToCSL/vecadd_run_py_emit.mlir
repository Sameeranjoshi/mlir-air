//===- vecadd_run_py_emit.mlir - HostEmitter FileCheck test -----*- MLIR -*-===//
//
// Part of the air-to-csl project.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
//
// Verifies that --emit-csl-rt --csl-output-dir writes a correct run.py
// matching the golden at mlir/test/Conversion/AIRToCSL/golden/run.py.golden.
//
// RUN: air-opt %s -csl-to-csl-rt | air-translate --emit-csl-rt --csl-output-dir=%t -o /dev/null
// RUN: cat %t/run.py | FileCheck %s
//
// CHECK: import argparse
// CHECK: import numpy as np
// CHECK: from cerebras.sdk.runtime.sdkruntimepybind import (
// CHECK: SdkRuntime
// CHECK: MemcpyDataType
// CHECK: MemcpyOrder
// CHECK: N = 256
// CHECK: parser = argparse.ArgumentParser
// CHECK: parser.add_argument("--name"
// CHECK: parser.add_argument("--cmaddr"
// CHECK: parser.add_argument("--check"
// CHECK: a = np.arange(N, dtype=np.float32)
// CHECK: b = np.arange(N, dtype=np.float32) * 2.0
// CHECK: c = np.zeros(N, dtype=np.float32)
// CHECK: expected = a + b
// CHECK: runner = SdkRuntime(args.name, cmaddr=args.cmaddr)
// CHECK-DAG: id_a = runner.get_id("a")
// CHECK-DAG: id_b = runner.get_id("b")
// CHECK-DAG: id_c = runner.get_id("c")
// CHECK: runner.load()
// CHECK: runner.run()
// CHECK: runner.memcpy_h2d(id_a, a, 0, 0, 1, 1, N
// CHECK: runner.memcpy_h2d(id_b, b, 0, 0, 1, 1, N
// CHECK: runner.launch("compute", nonblock=False)
// CHECK: runner.memcpy_d2h(c, id_c, 0, 0, 1, 1, N
// CHECK: runner.stop()
// CHECK: if args.check:
// CHECK: np.array_equal(c, expected)
// CHECK: print("PASS")
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
