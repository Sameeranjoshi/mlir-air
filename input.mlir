%kernel = csl.kernel "compute.csl" params({tile_id = 0 : i32}) {
  csl.var @data : memref<512xf32>
  csl.func @process() : () -> () {
    csl.return
  }
  csl.comptime {
    csl.export_symbol @data alias("input_buffer")
    csl.export_symbol @process
  }
} : !csl.kernel

csl.spatial_placement {
  %color = csl.color : !csl.color
  %route = csl.route in(RAMP) out(WEST) : i32

  %code_region = csl.code_region routes(%route) colors(%color) shape(8, 8) {
    csl.paint pe(0, 0) route(%route) color(%color)
  } {csl.params = ["width", 8 : i64, "M", 256 : i64, "N", 256 : i64]} : !csl.code_region

  csl.place %code_region at(0, 0) kernel(%kernel)
}
