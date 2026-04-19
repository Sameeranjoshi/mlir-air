// RUN: rm -rf %t && mkdir -p %t
// RUN: air-opt %s -csl-infer-exports | air-translate --emit-csl --output-dir=%t
// RUN: FileCheck %s --check-prefix=DEFAULT  < %t/sub_default/pe.csl
// RUN: FileCheck %s --check-prefix=OFFONLY  < %t/sub_offonly/pe.csl
// RUN: FileCheck %s --check-prefix=STRONLY  < %t/sub_stronly/pe.csl
// RUN: FileCheck %s --check-prefix=BOTH     < %t/sub_both/pe.csl
//
// Emitter coverage for csl.get_mem_dsd over memref.subview results. Verifies
// that default stride=1 / offset=0 are suppressed, non-defaults appear, and
// the expected @get_dsd(mem1d_dsd, ...) shape is emitted.

// DEFAULT-LABEL: fn compute() void
// DEFAULT: @get_dsd(mem1d_dsd, .{ .base_address = &x, .extent = 128 });
// DEFAULT-NOT: .stride
// DEFAULT-NOT: &x +

// OFFONLY-LABEL: fn compute() void
// OFFONLY: @get_dsd(mem1d_dsd, .{ .base_address = &x + 8, .extent = 64 });
// OFFONLY-NOT: .stride

// STRONLY-LABEL: fn compute() void
// STRONLY: @get_dsd(mem1d_dsd, .{ .base_address = &x, .extent = 32, .stride = 4 });
// STRONLY-NOT: &x +

// BOTH-LABEL: fn compute() void
// BOTH: @get_dsd(mem1d_dsd, .{ .base_address = &x + 3, .extent = 32, .stride = 4 });

module {
  // stride = 1, offset = 0 → both defaults collapse, no subview needed.
  csl.wafer @sub_default {arch = "wse3"} {
    csl.program @pe {
      %x = csl.var @x : memref<128xf32>
      csl.func @compute {
        %d = csl.get_mem_dsd %x : memref<128xf32> -> !csl.dsd
        csl.return
      }
      csl.export @x {alias = "x"}
      csl.export @compute {kind = "func"}
    }
    csl.layout {width = 1 : i64, height = 1 : i64} @layout {
      csl_layout.place @pe at (0, 0)
    }
    csl.host @main(%x_io: memref<128xf32>) {layout = @layout} {
      csl_host.memcpy_h2d %x_io to @layout::@x
          {px = 0 : i64, py = 0 : i64, width = 1 : i64, height = 1 : i64}
          : memref<128xf32>
      csl_host.launch @layout::@compute
    }
  }

  // offset = 8, stride = 1 (subview with nontrivial offset, default stride).
  csl.wafer @sub_offonly {arch = "wse3"} {
    csl.program @pe {
      %x = csl.var @x : memref<128xf32>
      csl.func @compute {
        %v = memref.subview %x[8] [64] [1]
             : memref<128xf32> to memref<64xf32, strided<[1], offset: 8>>
        %d = csl.get_mem_dsd %v
             : memref<64xf32, strided<[1], offset: 8>> -> !csl.dsd
        csl.return
      }
      csl.export @x {alias = "x"}
      csl.export @compute {kind = "func"}
    }
    csl.layout {width = 1 : i64, height = 1 : i64} @layout {
      csl_layout.place @pe at (0, 0)
    }
    csl.host @main(%x_io: memref<128xf32>) {layout = @layout} {
      csl_host.memcpy_h2d %x_io to @layout::@x
          {px = 0 : i64, py = 0 : i64, width = 1 : i64, height = 1 : i64}
          : memref<128xf32>
      csl_host.launch @layout::@compute
    }
  }

  // stride = 4, offset = 0.
  csl.wafer @sub_stronly {arch = "wse3"} {
    csl.program @pe {
      %x = csl.var @x : memref<128xf32>
      csl.func @compute {
        %v = memref.subview %x[0] [32] [4]
             : memref<128xf32> to memref<32xf32, strided<[4]>>
        %d = csl.get_mem_dsd %v : memref<32xf32, strided<[4]>> -> !csl.dsd
        csl.return
      }
      csl.export @x {alias = "x"}
      csl.export @compute {kind = "func"}
    }
    csl.layout {width = 1 : i64, height = 1 : i64} @layout {
      csl_layout.place @pe at (0, 0)
    }
    csl.host @main(%x_io: memref<128xf32>) {layout = @layout} {
      csl_host.memcpy_h2d %x_io to @layout::@x
          {px = 0 : i64, py = 0 : i64, width = 1 : i64, height = 1 : i64}
          : memref<128xf32>
      csl_host.launch @layout::@compute
    }
  }

  // stride = 4, offset = 3 — both non-default.
  csl.wafer @sub_both {arch = "wse3"} {
    csl.program @pe {
      %x = csl.var @x : memref<128xf32>
      csl.func @compute {
        %v = memref.subview %x[3] [32] [4]
             : memref<128xf32> to memref<32xf32, strided<[4], offset: 3>>
        %d = csl.get_mem_dsd %v
             : memref<32xf32, strided<[4], offset: 3>> -> !csl.dsd
        csl.return
      }
      csl.export @x {alias = "x"}
      csl.export @compute {kind = "func"}
    }
    csl.layout {width = 1 : i64, height = 1 : i64} @layout {
      csl_layout.place @pe at (0, 0)
    }
    csl.host @main(%x_io: memref<128xf32>) {layout = @layout} {
      csl_host.memcpy_h2d %x_io to @layout::@x
          {px = 0 : i64, py = 0 : i64, width = 1 : i64, height = 1 : i64}
          : memref<128xf32>
      csl_host.launch @layout::@compute
    }
  }
}
