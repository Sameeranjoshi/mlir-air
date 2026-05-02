// RUN: air-opt --csl-lower-dataflow-data %s | FileCheck %s

csl.wafer @w {arch = "wse3"} {
  csl.program @s {
    %a = csl.var @a : memref<8xf32>
    csl.func @c {
      %n = arith.constant 8 : index
      // CHECK: csl.builtin_call "fmovs"({{.*}}) {activate = @my_stream_put_done_0, async}
      csl.dataflow.put @my_stream source(%a) extent(%n : index) : memref<8xf32>
      csl.return
    }
    // CHECK: csl.task @my_stream_put_done_0
  }
  csl.layout {width = 2 : i64, height = 1 : i64} @layout {
    csl.color @my_stream_color {id = 0 : i32}
    csl_layout.dataflow @my_stream from(0, 0) to(1, 0) {color = @my_stream_color}
    csl_layout.set_color_config @my_stream_color at(0, 0) rx(RAMP) tx(EAST)
    csl_layout.set_color_config @my_stream_color at(1, 0) rx(WEST) tx(RAMP)
    csl_layout.place @s at (0, 0)
  }
}
