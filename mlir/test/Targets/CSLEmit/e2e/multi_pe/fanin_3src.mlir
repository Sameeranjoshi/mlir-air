// REQUIRES: cerebras-sdk
//
// RUN: rm -rf %t && mkdir -p %t
// RUN: air-opt --csl-dataflow-to-csl %s | air-translate --emit-csl --output-dir=%t
// RUN: cd %t/fanin_3src && bash commands_wse3.sh 2>&1 | FileCheck %s
//
// 4-PE fan-in: three sources send to one sink over separate channels.
//
//   pe0(0,0) ── ch_0 ──────────────▶ ┐
//   pe1(1,0) ──── ch_1 ──────────▶   ├──▶ pe3(3,0)
//   pe2(2,0) ──────── ch_2 ──────▶   ┘
//
// Channel routing (all EAST hops):
//   ch_0: from(0,0) to(3,0) — 3 hops (pe1, pe2 are pass-throughs for ch_0)
//   ch_1: from(1,0) to(3,0) — 2 hops (pe2 is a pass-through for ch_1)
//   ch_2: from(2,0) to(3,0) — 1 hop
//
// PE0, PE1, PE2 each have one put (N=1, no barrier needed).
// PE3 has three gets (N=3, triggers the multi-op barrier counter fix):
//   _barrier_ctr is initialised to 3 in compute(), each completion task
//   decrements it and only calls unblock_cmd_stream when it hits 0.
//
// H2D: in0 → pe0/buf, in1 → pe1/buf, in2 → pe2/buf (one arange each).
// D2H: pe3/buf0 → out0, pe3/buf1 → out1, pe3/buf2 → out2.
// Passthrough heuristic: 3 H2D == 3 D2H, fabric present →
//   verifies in0==out0, in1==out1, in2==out2 element-wise.
//
// CHECK: SUCCESS!

csl.wafer @fanin_3src {arch = "wse3"} {
  // --- PE 0: single put over ch_0 ---
  csl.program @pe0 {
    %buf = csl.var @buf : memref<64xf32>
    csl.func @compute {
      %n = arith.constant 64 : index
      csl.dataflow.put @ch_0 source(%buf) extent(%n : index) : memref<64xf32>
      csl.return
    }
    csl.export @buf {alias = "buf_pe0_in", direction = "in"}
    csl.export @compute {kind = "func", direction = "internal"}
  }

  // --- PE 1: single put over ch_1 ---
  csl.program @pe1 {
    %buf = csl.var @buf : memref<64xf32>
    csl.func @compute {
      %n = arith.constant 64 : index
      csl.dataflow.put @ch_1 source(%buf) extent(%n : index) : memref<64xf32>
      csl.return
    }
    csl.export @buf {alias = "buf_pe1_in", direction = "in"}
    csl.export @compute {kind = "func", direction = "internal"}
  }

  // --- PE 2: single put over ch_2 ---
  csl.program @pe2 {
    %buf = csl.var @buf : memref<64xf32>
    csl.func @compute {
      %n = arith.constant 64 : index
      csl.dataflow.put @ch_2 source(%buf) extent(%n : index) : memref<64xf32>
      csl.return
    }
    csl.export @buf {alias = "buf_pe2_in", direction = "in"}
    csl.export @compute {kind = "func", direction = "internal"}
  }

  // --- PE 3: fan-in sink — receives from ch_0, ch_1, ch_2 (N=3 gets) ---
  csl.program @pe3 {
    %buf0 = csl.var @buf0 : memref<64xf32>
    %buf1 = csl.var @buf1 : memref<64xf32>
    %buf2 = csl.var @buf2 : memref<64xf32>
    csl.func @compute {
      %n = arith.constant 64 : index
      csl.dataflow.get @ch_0 target(%buf0) extent(%n : index) : memref<64xf32>
      csl.dataflow.get @ch_1 target(%buf1) extent(%n : index) : memref<64xf32>
      csl.dataflow.get @ch_2 target(%buf2) extent(%n : index) : memref<64xf32>
      csl.return
    }
    csl.export @buf0 {alias = "buf_pe3_0", direction = "out"}
    csl.export @buf1 {alias = "buf_pe3_1", direction = "out"}
    csl.export @buf2 {alias = "buf_pe3_2", direction = "out"}
    csl.export @compute {kind = "func", direction = "internal"}
  }

  // --- Layout: 4×1 grid ---
  csl.layout {width = 4 : i64, height = 1 : i64} @layout {
    // Three separate channels, all flowing EAST to pe3 at (3,0).
    csl_layout.dataflow @ch_0 from(0, 0) to(3, 0)
    csl_layout.dataflow @ch_1 from(1, 0) to(3, 0)
    csl_layout.dataflow @ch_2 from(2, 0) to(3, 0)

    // Place programs
    csl_layout.place @pe0 at (0, 0)
    csl_layout.place @pe1 at (1, 0)
    csl_layout.place @pe2 at (2, 0)
    csl_layout.place @pe3 at (3, 0)

    // Export input buffers (H2D targets)
    csl_layout.export "buf_pe0_in" from @pe0::@buf
    csl_layout.export "buf_pe1_in" from @pe1::@buf
    csl_layout.export "buf_pe2_in" from @pe2::@buf

    // Export output buffers (D2H sources)
    csl_layout.export "buf_pe3_0" from @pe3::@buf0
    csl_layout.export "buf_pe3_1" from @pe3::@buf1
    csl_layout.export "buf_pe3_2" from @pe3::@buf2

    // Export compute entry point (from pe0 by convention; host calls it
    // width-broadcast so all PEs execute compute simultaneously).
    csl_layout.export "compute" from @pe0::@compute {kind = "func"}
  }

  // --- Host driver ---
  csl.host @main(%in0: memref<64xf32>, %in1: memref<64xf32>,
                 %in2: memref<64xf32>,
                 %out0: memref<64xf32>, %out1: memref<64xf32>,
                 %out2: memref<64xf32>) {layout = @layout} {
    // H2D: send inputs to the three source PEs.
    csl_host.memcpy_h2d %in0 to @layout::@buf_pe0_in
        {px = 0 : i64, py = 0 : i64, width = 1 : i64, height = 1 : i64}
        : memref<64xf32>
    csl_host.memcpy_h2d %in1 to @layout::@buf_pe1_in
        {px = 1 : i64, py = 0 : i64, width = 1 : i64, height = 1 : i64}
        : memref<64xf32>
    csl_host.memcpy_h2d %in2 to @layout::@buf_pe2_in
        {px = 2 : i64, py = 0 : i64, width = 1 : i64, height = 1 : i64}
        : memref<64xf32>

    // Launch compute on all PEs simultaneously.
    csl_host.launch @layout::@compute

    // D2H: collect results from the sink PE.
    csl_host.memcpy_d2h @layout::@buf_pe3_0 to %out0
        {px = 3 : i64, py = 0 : i64, width = 1 : i64, height = 1 : i64}
        : memref<64xf32>
    csl_host.memcpy_d2h @layout::@buf_pe3_1 to %out1
        {px = 3 : i64, py = 0 : i64, width = 1 : i64, height = 1 : i64}
        : memref<64xf32>
    csl_host.memcpy_d2h @layout::@buf_pe3_2 to %out2
        {px = 3 : i64, py = 0 : i64, width = 1 : i64, height = 1 : i64}
        : memref<64xf32>
  }
}
