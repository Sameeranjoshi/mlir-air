// REQUIRES: cerebras-sdk
//
// RUN: rm -rf %t && mkdir -p %t
// RUN: air-opt --csl-dataflow-to-csl %s | air-translate --emit-csl --output-dir=%t
// RUN: cd %t/reduce_chain_4pe && bash commands_wse3.sh 2>&1 | FileCheck %s
//
// Reduction chain: P0 (source, 100) → P1 (relay, -100) → P2 (relay, +100) → P3 (sink).
//
//   P0 sends 100.0 per element.
//   P1 receives, subtracts 100.0 → 0.0, forwards.
//   P2 receives, adds 100.0 → 100.0, forwards.
//   P3 receives final value 100.0.
//
//   With 2 relays (even count), the alternating ±100 operations cancel:
//     100 - 100 + 100 = 100  →  P3 should equal the source (P0).
//
// The passthrough heuristic fires (1 H2D, 1 D2H, fabric ops present)
// and verifies output == input element-wise.
//
// CHECK: SUCCESS!

csl.wafer @reduce_chain_4pe {arch = "wse3"} {
  csl.program @p0 {
    %buf = csl.var @buf : memref<64xf32>
    csl.func @compute {
      %n = arith.constant 64 : index
      csl.dataflow.put @ch01 source(%buf) extent(%n : index) : memref<64xf32>
      csl.return
    }
    csl.export @buf {alias = "buf_p0", direction = "in"}
    csl.export @compute {kind = "func", direction = "internal"}
  }
  csl.program @p1 {
    %buf = csl.var @buf : memref<64xf32>
    csl.func @compute {
      %n     = arith.constant 64   : index
      %c0    = arith.constant 0    : index
      %c1    = arith.constant 1    : index
      %delta = arith.constant -100.0 : f32
      csl.dataflow.get @ch01 target(%buf) extent(%n : index) : memref<64xf32>
      scf.for %i = %c0 to %n step %c1 {
        %v  = memref.load  %buf[%i] : memref<64xf32>
        %nv = arith.addf %v, %delta : f32
        memref.store %nv, %buf[%i] : memref<64xf32>
      }
      csl.dataflow.put @ch12 source(%buf) extent(%n : index) : memref<64xf32>
      csl.return
    }
    csl.export @compute {kind = "func", direction = "internal"}
  }
  csl.program @p2 {
    %buf = csl.var @buf : memref<64xf32>
    csl.func @compute {
      %n     = arith.constant 64  : index
      %c0    = arith.constant 0   : index
      %c1    = arith.constant 1   : index
      %delta = arith.constant 100.0 : f32
      csl.dataflow.get @ch12 target(%buf) extent(%n : index) : memref<64xf32>
      scf.for %i = %c0 to %n step %c1 {
        %v  = memref.load  %buf[%i] : memref<64xf32>
        %nv = arith.addf %v, %delta : f32
        memref.store %nv, %buf[%i] : memref<64xf32>
      }
      csl.dataflow.put @ch23 source(%buf) extent(%n : index) : memref<64xf32>
      csl.return
    }
    csl.export @compute {kind = "func", direction = "internal"}
  }
  csl.program @p3 {
    %buf = csl.var @buf : memref<64xf32>
    csl.func @compute {
      %n = arith.constant 64 : index
      csl.dataflow.get @ch23 target(%buf) extent(%n : index) : memref<64xf32>
      csl.return
    }
    csl.export @buf {alias = "buf_p3", direction = "out"}
    csl.export @compute {kind = "func", direction = "internal"}
  }
  csl.layout {width = 4 : i64, height = 1 : i64} @layout {
    csl_layout.dataflow @ch01 from(0, 0) to(1, 0)
    csl_layout.dataflow @ch12 from(1, 0) to(2, 0)
    csl_layout.dataflow @ch23 from(2, 0) to(3, 0)
    csl_layout.place @p0 at (0, 0)
    csl_layout.place @p1 at (1, 0)
    csl_layout.place @p2 at (2, 0)
    csl_layout.place @p3 at (3, 0)
    csl_layout.export "buf_p0"  from @p0::@buf
    csl_layout.export "buf_p3"  from @p3::@buf
    csl_layout.export "compute" from @p0::@compute {kind = "func"}
  }
  csl.host @main(%in: memref<64xf32>, %out: memref<64xf32>) {layout = @layout} {
    csl_host.memcpy_h2d %in to @layout::@buf_p0
        {px = 0 : i64, py = 0 : i64, width = 1 : i64, height = 1 : i64}
        : memref<64xf32>
    csl_host.launch @layout::@compute
    csl_host.memcpy_d2h @layout::@buf_p3 to %out
        {px = 3 : i64, py = 0 : i64, width = 1 : i64, height = 1 : i64}
        : memref<64xf32>
  }
}
