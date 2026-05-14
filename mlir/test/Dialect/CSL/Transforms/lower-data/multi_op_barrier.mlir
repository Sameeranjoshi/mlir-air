// RUN: air-opt --csl-lower-dataflow-data %s | FileCheck %s
//
// When a program has N > 1 stream ops the pass must:
//   1. Insert csl.var @_barrier_ctr : memref<1xi16> at program scope.
//   2. Initialise the counter to N at the start of csl.func @compute.
//   3. Add {barrier_total = N : i32} to every completion task.
//   4. NOT emit unblock_cmd_stream inside the task body (the emitter
//      will synthesise the countdown idiom instead).
//
// This test uses 2 puts in one program so N == 2.

// CHECK-LABEL: csl.program @sender
// CHECK:       csl.var @_barrier_ctr : memref<1xi16>

csl.wafer @w {arch = "wse3"} {
  csl.program @sender {
    %a = csl.var @a : memref<32xf32>
    %b = csl.var @b : memref<32xf32>
    csl.func @compute {
      %n = arith.constant 32 : index

      // CHECK: arith.constant 2 : i16
      // CHECK: memref.store

      csl.dataflow.put @ch1 source(%a) extent(%n : index) : memref<32xf32>
      csl.dataflow.put @ch2 source(%b) extent(%n : index) : memref<32xf32>
      csl.return
    }
    // Tasks must have barrier_total = 2 and must NOT contain unblock_cmd_stream.
    // CHECK-DAG: csl.task @ch1_put_done_0 attributes {barrier_total = 2 : i32, id = 8 : i32, trigger_kind = "local_task_id"}
    // CHECK-DAG: csl.task @ch2_put_done_1 attributes {barrier_total = 2 : i32, id = 9 : i32, trigger_kind = "local_task_id"}
    // CHECK-NOT: csl.builtin_call "unblock_cmd_stream"
  }
  csl.layout {width = 2 : i64, height = 2 : i64} @layout {
    csl.color @ch1_color {id = 0 : i32}
    csl.color @ch2_color {id = 1 : i32}
    csl_layout.dataflow @ch1 from(0, 0) to(1, 0) {color = @ch1_color}
    csl_layout.dataflow @ch2 from(0, 0) to(0, 1) {color = @ch2_color}
    csl_layout.set_color_config @ch1_color at(0, 0) rx(RAMP) tx(EAST)
    csl_layout.set_color_config @ch1_color at(1, 0) rx(WEST) tx(RAMP)
    csl_layout.set_color_config @ch2_color at(0, 0) rx(RAMP) tx(SOUTH)
    csl_layout.set_color_config @ch2_color at(0, 1) rx(NORTH) tx(RAMP)
    csl_layout.place @sender at (0, 0)
  }
}
