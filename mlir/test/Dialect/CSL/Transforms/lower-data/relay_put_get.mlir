// RUN: air-opt --csl-lower-dataflow-data %s | FileCheck %s

// A relay PE: one get, one scf.for transform, one put. The pass should
// generate a relay task (activated by the get) that contains the transform
// and the put's async fmovs. A separate put_done task calls unblock_cmd_stream.

csl.wafer @w {arch = "wse3"} {
  csl.program @relay_pe {
    %buf = csl.var @buf : memref<64xf32>
    csl.func @compute {
      %n     = arith.constant 64   : index
      %c0    = arith.constant 0    : index
      %c1    = arith.constant 1    : index
      %delta = arith.constant -100.0 : f32
      csl.dataflow.get @ch_in  target(%buf) extent(%n : index) : memref<64xf32>
      scf.for %i = %c0 to %n step %c1 {
        %v  = memref.load  %buf[%i] : memref<64xf32>
        %nv = arith.addf %v, %delta : f32
        memref.store %nv, %buf[%i] : memref<64xf32>
      }
      csl.dataflow.put @ch_out source(%buf) extent(%n : index) : memref<64xf32>
      csl.return
    }
  }
  csl.layout {width = 3 : i64, height = 1 : i64} @layout {
    csl.color @ch_in_color  {id = 0 : i32}
    csl.color @ch_out_color {id = 1 : i32}
    csl_layout.dataflow @ch_in  from(0, 0) to(1, 0) {color = @ch_in_color}
    csl_layout.dataflow @ch_out from(1, 0) to(2, 0) {color = @ch_out_color}
    csl_layout.set_color_config @ch_in_color  at(1, 0) rx(WEST) tx(RAMP)
    csl_layout.set_color_config @ch_out_color at(1, 0) rx(RAMP) tx(EAST)
    csl_layout.place @relay_pe at (1, 0)
  }
}

// After lowering, compute() should have the GET fmovs activating _relay_0.
// CHECK: csl.func @compute
// CHECK:   csl.get_mem_dsd %{{.*}} : memref<64xf32> -> !csl.dsd
// CHECK:   csl.get_fab_dsd fabin @ch_in_color extent(%{{.*}} : index) : !csl.dsd
// CHECK:   csl.builtin_call "fmovs"({{.*}}) {activate = @_relay_0, async}

// The relay task should exist with id=8, contain the scf.for transform, and
// emit the PUT fmovs activating ch_out_put_done_1.
// CHECK: csl.task @_relay_0 attributes {id = 8 : i32, trigger_kind = "local_task_id"}
// CHECK:   scf.for
// CHECK:   csl.get_mem_dsd %{{.*}} : memref<64xf32> -> !csl.dsd
// CHECK:   csl.get_fab_dsd fabout @ch_out_color extent(%{{.*}} : index) : !csl.dsd
// CHECK:   csl.builtin_call "fmovs"({{.*}}) {activate = @ch_out_put_done_1, async}

// The put_done task should call unblock_cmd_stream.
// CHECK: csl.task @ch_out_put_done_1 attributes {id = 9 : i32, trigger_kind = "local_task_id"}
// CHECK:   csl.builtin_call "unblock_cmd_stream"()
