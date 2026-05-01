// RUN: air-opt --csl-lower-stream-data %s | FileCheck %s

// CHECK-NOT: csl_layout.stream
// CHECK-NOT: csl.stream.put
// CHECK-NOT: csl.stream.get

csl.wafer @w {arch = "wse3"} {
  csl.program @left {
    %buf = csl.var @buf : memref<32xf32>
    csl.func @c {
      %n = arith.constant 32 : index
      csl.stream.put @ch source(%buf) extent(%n : index) : memref<32xf32>
      csl.return
    }
  }
  csl.layout {width = 2 : i64, height = 1 : i64} @layout {
    csl.color @ch_color {id = 0 : i32}
    csl_layout.stream @ch from(0, 0) to(1, 0) {color = @ch_color}
    csl_layout.set_color_config @ch_color at(0, 0) rx(RAMP) tx(EAST)
    csl_layout.set_color_config @ch_color at(1, 0) rx(WEST) tx(RAMP)
    csl_layout.place @left at (0, 0)
  }
}
