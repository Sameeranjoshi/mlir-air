// REQUIRES: cerebras-sdk
//
// RUN: rm -rf %t && mkdir -p %t
// RUN: air-opt --csl-dataflow-to-csl %s | air-translate --emit-csl --output-dir=%t
// RUN: cd %t/reduce_chain_4pe_wavelet && bash commands_wse3.sh 2>&1 | FileCheck %s
//
// Wavelet / data-task reduction chain.  Each relay PE binds a CSL *data task*
// to its input color; the task fires once per incoming wavelet, receives the
// payload as a block argument, modifies it, and forwards it on the output
// color via a synchronous 1-element @fmovs (csl.dataflow.send_wavelet).
//
//   P0 (source, bulk fmovs 64 f32) → P1 (data task, +(-100)) →
//   P2 (data task, +(+100))        → P3 (sink, bulk fmovs 64 f32)
//
//   100 − 100 + 100 = 100 → output equals input.
//
// The passthrough heuristic fires (1 H2D, 1 D2H, fabric ops present) and
// verifies output == input element-wise.
//
// CHECK: SUCCESS!

csl.wafer @reduce_chain_4pe_wavelet {arch = "wse3"} {
  // P0: source — fill buf with 100.0, bulk-send 64 wavelets to P1.
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

  // P1: data task relay — fires once per wavelet on @ch01.
  //     Subtracts 100 from each wavelet payload and forwards on @ch12.
  csl.program @p1 {
    csl.func @compute {
      csl.return
    }
    csl.task @relay attributes {trigger_kind = "data_task", color = @ch01} {
    ^bb0(%val: f32):
      %delta = arith.constant -100.0 : f32
      %nv = arith.addf %val, %delta : f32
      csl.dataflow.send_wavelet @ch12 value(%nv) : f32
      csl.return
    }
    csl.export @compute {kind = "func", direction = "internal"}
  }

  // P2: data task relay — fires once per wavelet on @ch12.
  //     Adds 100 to each wavelet payload and forwards on @ch23.
  csl.program @p2 {
    csl.func @compute {
      csl.return
    }
    csl.task @relay attributes {trigger_kind = "data_task", color = @ch12} {
    ^bb0(%val: f32):
      %delta = arith.constant 100.0 : f32
      %nv = arith.addf %val, %delta : f32
      csl.dataflow.send_wavelet @ch23 value(%nv) : f32
      csl.return
    }
    csl.export @compute {kind = "func", direction = "internal"}
  }

  // P3: sink — receives 64 wavelets from @ch23 into buf.
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
