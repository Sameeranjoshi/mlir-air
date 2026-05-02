// REQUIRES: cerebras-sdk
//
// RUN: rm -rf %t && mkdir -p %t
// RUN: air-opt --csl-dataflow-to-csl %s | air-translate --emit-csl --output-dir=%t
// RUN: cd %t/reduce_chain_4pe_xparity && bash commands_wse3.sh 2>&1 | FileCheck %s
//
// Reduction chain where each relay's behavior is decided by its x coord:
//   even x: add  100      odd x: subtract 100
// Both relay programs (@p1_relay at x=1, @p2_relay at x=2) have IDENTICAL
// code; the runtime layout_mod.get_x_coord() picks the operation. This
// demonstrates how a single relay shape can run across heterogeneous PEs.
//
//   P0 (source: 100) -> P1 (x=1, odd: -100 -> 0) -> P2 (x=2, even: +100 -> 100)
//   -> P3 (sink, receives 100). Net: 100 - 100 + 100 = 100.
//
// CHECK: SUCCESS!

csl.wafer @reduce_chain_4pe_xparity {arch = "wse3"} {
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

  // Relay program for x=1 (forwards on @ch12). Body branches on x parity.
  // Uses index type (maps to u16) to match layout_mod.get_x_coord() return type.
  csl.program @p1_relay {
    csl.func @compute { csl.return }
    csl.task @relay attributes {trigger_kind = "data_task", color = @ch01} {
    ^bb0(%val: f32):
      %x_i16   = csl.get_x_coord : i16
      %x       = arith.index_cast %x_i16 : i16 to index
      %two     = arith.constant 2 : index
      %r       = arith.remui %x, %two : index
      %zero    = arith.constant 0 : index
      %is_even = arith.cmpi eq, %r, %zero : index
      %pos100  = arith.constant 100.0  : f32
      %neg100  = arith.constant -100.0 : f32
      %delta   = arith.select %is_even, %pos100, %neg100 : f32
      %nv      = arith.addf %val, %delta : f32
      csl.dataflow.send_wavelet @ch12 value(%nv) : f32
      csl.return
    }
    csl.export @compute {kind = "func", direction = "internal"}
  }

  // Relay program for x=2 (forwards on @ch23). Body is identical to p1_relay.
  csl.program @p2_relay {
    csl.func @compute { csl.return }
    csl.task @relay attributes {trigger_kind = "data_task", color = @ch12} {
    ^bb0(%val: f32):
      %x_i16   = csl.get_x_coord : i16
      %x       = arith.index_cast %x_i16 : i16 to index
      %two     = arith.constant 2 : index
      %r       = arith.remui %x, %two : index
      %zero    = arith.constant 0 : index
      %is_even = arith.cmpi eq, %r, %zero : index
      %pos100  = arith.constant 100.0  : f32
      %neg100  = arith.constant -100.0 : f32
      %delta   = arith.select %is_even, %pos100, %neg100 : f32
      %nv      = arith.addf %val, %delta : f32
      csl.dataflow.send_wavelet @ch23 value(%nv) : f32
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
    csl_layout.place @p0       at (0, 0)
    csl_layout.place @p1_relay at (1, 0)
    csl_layout.place @p2_relay at (2, 0)
    csl_layout.place @p3       at (3, 0)
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
