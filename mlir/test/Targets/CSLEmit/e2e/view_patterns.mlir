// RUN: rm -rf %t && mkdir -p %t
// RUN: air-opt %s -csl-infer-exports | air-translate --emit-csl --output-dir=%t
// RUN: FileCheck %s --check-prefix=DEFAULT  < %t/view_default/pe.csl
// RUN: FileCheck %s --check-prefix=OFFONLY  < %t/view_offonly/pe.csl
// RUN: FileCheck %s --check-prefix=STRONLY  < %t/view_stronly/pe.csl
// RUN: FileCheck %s --check-prefix=BOTH     < %t/view_both/pe.csl
// RUN: FileCheck %s --check-prefix=DYNAMIC  < %t/view_dynamic/pe.csl
//
// Emitter-coverage for the strided view: verifies that default
// stride=1 / offset=0 are suppressed, non-defaults appear, and dynamic
// SSA values (not arith.constant) emit as variable references.

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

// DYNAMIC-LABEL: fn compute() void
// The stride here is loaded from a runtime memref, so it should emit as a
// CSL variable name rather than a literal integer.
// DYNAMIC: @get_dsd(mem1d_dsd, .{ .base_address = &x, .extent = 32, .stride =
// DYNAMIC-NOT: .stride = 1 });
// DYNAMIC-NOT: .stride = 2 });

module {
  // stride = 1, offset = 0 — both defaults collapse.
  csl.wafer @view_default {arch = "wse3"} {
    csl.program @pe {
      %x = csl.var @x : memref<128xf32>
      csl.func @compute {
        %n   = arith.constant 128 : index
        %ext = arith.constant 128 : index
        %str = arith.constant   1 : index
        %off = arith.constant   0 : index
        %v   = csl.view.strided %ext, %str, %off : !csl.view
        %d   = csl.get_mem_dsd %x, %n view %v
                 : memref<128xf32>, index, !csl.view -> !csl.dsd
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

  // offset != 0, stride = 1.
  csl.wafer @view_offonly {arch = "wse3"} {
    csl.program @pe {
      %x = csl.var @x : memref<128xf32>
      csl.func @compute {
        %n   = arith.constant 128 : index
        %ext = arith.constant  64 : index
        %str = arith.constant   1 : index
        %off = arith.constant   8 : index
        %v   = csl.view.strided %ext, %str, %off : !csl.view
        %d   = csl.get_mem_dsd %x, %n view %v
                 : memref<128xf32>, index, !csl.view -> !csl.dsd
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

  // stride != 1, offset = 0.
  csl.wafer @view_stronly {arch = "wse3"} {
    csl.program @pe {
      %x = csl.var @x : memref<128xf32>
      csl.func @compute {
        %n   = arith.constant 128 : index
        %ext = arith.constant  32 : index
        %str = arith.constant   4 : index
        %off = arith.constant   0 : index
        %v   = csl.view.strided %ext, %str, %off : !csl.view
        %d   = csl.get_mem_dsd %x, %n view %v
                 : memref<128xf32>, index, !csl.view -> !csl.dsd
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

  // Both stride and offset non-default.
  csl.wafer @view_both {arch = "wse3"} {
    csl.program @pe {
      %x = csl.var @x : memref<128xf32>
      csl.func @compute {
        %n   = arith.constant 128 : index
        %ext = arith.constant  32 : index
        %str = arith.constant   4 : index
        %off = arith.constant   3 : index
        %v   = csl.view.strided %ext, %str, %off : !csl.view
        %d   = csl.get_mem_dsd %x, %n view %v
                 : memref<128xf32>, index, !csl.view -> !csl.dsd
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

  // Dynamic stride: loaded from a host-visible memref at runtime, then
  // index_cast → feeds the view directly. The emitter should write the
  // SSA name (not a literal integer) into `.stride = ...`.
  csl.wafer @view_dynamic {arch = "wse3"} {
    csl.program @pe {
      %x   = csl.var @x   : memref<128xf32>
      %cfg = csl.var @cfg : memref<1xi32>
      csl.func @compute {
        %c0  = arith.constant 0 : index
        %n   = arith.constant 128 : index
        %ext = arith.constant  32 : index
        %off = arith.constant   0 : index
        %si  = memref.load %cfg[%c0] : memref<1xi32>
        %str = arith.index_cast %si : i32 to index
        %v   = csl.view.strided %ext, %str, %off : !csl.view
        %d   = csl.get_mem_dsd %x, %n view %v
                 : memref<128xf32>, index, !csl.view -> !csl.dsd
        csl.return
      }
      csl.export @x   {alias = "x"}
      csl.export @cfg {alias = "cfg"}
      csl.export @compute {kind = "func"}
    }
    csl.layout {width = 1 : i64, height = 1 : i64} @layout {
      csl_layout.place @pe at (0, 0)
    }
    csl.host @main(%x_io: memref<128xf32>, %cfg_in: memref<1xi32>)
        {layout = @layout} {
      csl_host.memcpy_h2d %x_io to @layout::@x
          {px = 0 : i64, py = 0 : i64, width = 1 : i64, height = 1 : i64}
          : memref<128xf32>
      csl_host.memcpy_h2d %cfg_in to @layout::@cfg
          {px = 0 : i64, py = 0 : i64, width = 1 : i64, height = 1 : i64}
          : memref<1xi32>
      csl_host.launch @layout::@compute
    }
  }
}
