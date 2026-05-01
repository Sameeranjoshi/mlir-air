// REQUIRES: cerebras-sdk
//
// RUN: rm -rf %t && mkdir -p %t
// RUN: air-opt --csl-streams-to-csl %s | air-translate --emit-csl --output-dir=%t
// RUN: cd %t/ping_2pe && bash commands_wse3.sh 2>&1 | FileCheck %s
//
// MILESTONE: 2-PE inter-PE dataflow via a CSL fabric stream, end-to-end on the
// WSE-3 simulator.
//
//   left_pe ─── send_ch ──▶ right_pe
//   (0,0)                   (1,0)
//
// `csl.stream.put` on the producer side and `csl.stream.get` on the consumer
// side are lowered by the --csl-streams-to-csl pipeline into a fabric DSD +
// async fmovs + completion task triplet. The emitter then writes one
// per-program .csl file, a layout.csl wrapping the placement and routing,
// and a host run.py that does:
//   memcpy_h2d arg0 -> @left_pe::@buf
//   launch compute (fires on both PEs in parallel)
//   memcpy_d2h @right_pe::@buf -> arg1
// The host then asserts arg0 == arg1 elementwise (the fabric pass-through
// heuristic in CSLHostEmitter).
//
// CHECK: SUCCESS!

csl.wafer @ping_2pe {arch = "wse3"} {
  csl.program @left_pe {
    %buf = csl.var @buf : memref<128xf32>
    csl.func @compute {
      %n = arith.constant 128 : index
      csl.stream.put @send_ch source(%buf) extent(%n : index) : memref<128xf32>
      csl.return
    }
    csl.export @buf {alias = "buf_left", direction = "in"}
    csl.export @compute {kind = "func", direction = "internal"}
  }
  csl.program @right_pe {
    %buf = csl.var @buf : memref<128xf32>
    csl.func @compute {
      %n = arith.constant 128 : index
      csl.stream.get @send_ch target(%buf) extent(%n : index) : memref<128xf32>
      csl.return
    }
    csl.export @buf {alias = "buf_right", direction = "out"}
    csl.export @compute {kind = "func", direction = "internal"}
  }
  csl.layout {width = 2 : i64, height = 1 : i64} @layout {
    csl_layout.stream @send_ch from(0, 0) to(1, 0)
    csl_layout.place  @left_pe  at (0, 0)
    csl_layout.place  @right_pe at (1, 0)
    csl_layout.export "buf_left"  from @left_pe::@buf
    csl_layout.export "buf_right" from @right_pe::@buf
    csl_layout.export "compute"   from @left_pe::@compute  {kind = "func"}
  }
  csl.host @main(%in: memref<128xf32>, %out: memref<128xf32>) {layout = @layout} {
    csl_host.memcpy_h2d %in to @layout::@buf_left
        {px = 0 : i64, py = 0 : i64, width = 1 : i64, height = 1 : i64}
        : memref<128xf32>
    csl_host.launch @layout::@compute
    csl_host.memcpy_d2h @layout::@buf_right to %out
        {px = 1 : i64, py = 0 : i64, width = 1 : i64, height = 1 : i64}
        : memref<128xf32>
  }
}
