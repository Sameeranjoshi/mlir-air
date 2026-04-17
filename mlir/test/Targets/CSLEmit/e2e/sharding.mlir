// RUN: rm -rf %t && mkdir -p %t
// RUN: air-opt %s -csl-infer-exports | air-translate --emit-csl --output-dir=%t
// RUN: FileCheck --check-prefix=POINT %s < %t/shard_point/run.py
// RUN: FileCheck --check-prefix=ROW   %s < %t/shard_row/run.py
// RUN: FileCheck --check-prefix=GRID  %s < %t/shard_grid/run.py
//
// Task 7: host emitter derives (w, h, l) from csl_layout.place extent.
//   * shard_point: 256-elem memref on a 1x1 placement -> (1, 1, 256)
//   * shard_row:   256-elem memref on a 8x1 placement -> (8, 1, 32)
//   * shard_grid:  16x16 memref on a 4x4 placement    -> (4, 4, 16)

// POINT: runner.memcpy_h2d(runner.get_id("a"), arg0, 0, 0, 1, 1, 256,
// POINT: runner.memcpy_d2h(arg1, runner.get_id("b"), 0, 0, 1, 1, 256,

// ROW: runner.memcpy_h2d(runner.get_id("a"), arg0, 0, 0, 8, 1, 32,
// ROW: runner.memcpy_d2h(arg1, runner.get_id("b"), 0, 0, 8, 1, 32,

// GRID: runner.memcpy_h2d(runner.get_id("a"), arg0, 0, 0, 4, 4, 16,
// GRID: runner.memcpy_d2h(arg1, runner.get_id("b"), 0, 0, 4, 4, 16,

module {
  // 1x1 placement, 256-elem buffer: each PE gets the full 256 elements.
  csl.wafer @shard_point {arch = "wse3"} {
    csl.program @pe {
      %a = csl.var @a : memref<256xf32>
      %b = csl.var @b : memref<256xf32>
      csl.func @compute { csl.return }
      csl.export @a {alias = "a"}
      csl.export @b {alias = "b"}
      csl.export @compute {kind = "func"}
    }
    csl.layout {width = 1 : i64, height = 1 : i64} @layout {
      csl_layout.place @pe at (0, 0)
    }
    csl.host @main(%a_in: memref<256xf32>, %b_out: memref<256xf32>)
        {layout = @layout} {
      csl_host.memcpy_h2d %a_in to @layout::@a
          {px = 0 : i64, py = 0 : i64, width = 1 : i64, height = 1 : i64}
          : memref<256xf32>
      csl_host.launch @layout::@compute
      csl_host.memcpy_d2h @layout::@b to %b_out
          {px = 0 : i64, py = 0 : i64, width = 1 : i64, height = 1 : i64}
          : memref<256xf32>
    }
  }

  // 8x1 placement, 256-elem buffer: each PE gets 32 elements.
  csl.wafer @shard_row {arch = "wse3"} {
    csl.program @pe {
      %a = csl.var @a : memref<32xf32>
      %b = csl.var @b : memref<32xf32>
      csl.func @compute { csl.return }
      csl.export @a {alias = "a"}
      csl.export @b {alias = "b"}
      csl.export @compute {kind = "func"}
    }
    csl.layout {width = 8 : i64, height = 1 : i64} @layout {
      csl_layout.place @pe over [0:8, 0]
    }
    csl.host @main(%a_in: memref<256xf32>, %b_out: memref<256xf32>)
        {layout = @layout} {
      csl_host.memcpy_h2d %a_in to @layout::@a
          {px = 0 : i64, py = 0 : i64, width = 8 : i64, height = 1 : i64}
          : memref<256xf32>
      csl_host.launch @layout::@compute
      csl_host.memcpy_d2h @layout::@b to %b_out
          {px = 0 : i64, py = 0 : i64, width = 8 : i64, height = 1 : i64}
          : memref<256xf32>
    }
  }

  // 4x4 placement, 16x16 buffer: each PE gets 16 elements.
  csl.wafer @shard_grid {arch = "wse3"} {
    csl.program @pe {
      %a = csl.var @a : memref<16xf32>
      %b = csl.var @b : memref<16xf32>
      csl.func @compute { csl.return }
      csl.export @a {alias = "a"}
      csl.export @b {alias = "b"}
      csl.export @compute {kind = "func"}
    }
    csl.layout {width = 4 : i64, height = 4 : i64} @layout {
      csl_layout.place @pe over [0:4, 0:4]
    }
    csl.host @main(%a_in: memref<16x16xf32>, %b_out: memref<16x16xf32>)
        {layout = @layout} {
      csl_host.memcpy_h2d %a_in to @layout::@a
          {px = 0 : i64, py = 0 : i64, width = 4 : i64, height = 4 : i64}
          : memref<16x16xf32>
      csl_host.launch @layout::@compute
      csl_host.memcpy_d2h @layout::@b to %b_out
          {px = 0 : i64, py = 0 : i64, width = 4 : i64, height = 4 : i64}
          : memref<16x16xf32>
    }
  }
}
