// RUN: rm -rf %t && mkdir -p %t
// RUN: air-opt %s -csl-infer-exports | air-translate --emit-csl --output-dir=%t
// RUN: FileCheck --check-prefix=WA %s < %t/wa/pe_a.csl
// RUN: FileCheck --check-prefix=WB %s < %t/wb/pe_b.csl
// RUN: test -f %t/wa/commands_wse3.sh
// RUN: test -f %t/wb/commands_wse3.sh
//
// Multi-wafer integration test. Two wafers in one module must each get their
// own output subdirectory.

module {
  csl.wafer @wa {arch = "wse3"} {
    csl.program @pe_a {
      %x = csl.var @x : memref<16xf32>
      csl.func @compute {
        csl.return
      }
      csl.export @x {alias = "x"}
      csl.export @compute {kind = "func"}
    }
    csl.layout {width = 1 : i64, height = 1 : i64} @layout {
      csl_layout.place @pe_a at (0, 0)
    }
    csl.host @main(%x: memref<16xf32>) {layout = @layout} {
      csl_host.memcpy_h2d %x to @layout::@x
          {px = 0 : i64, py = 0 : i64, width = 1 : i64, height = 1 : i64}
          : memref<16xf32>
      csl_host.launch @layout::@compute
    }
  }

  csl.wafer @wb {arch = "wse3"} {
    csl.program @pe_b {
      %y = csl.var @y : memref<32xf32>
      csl.func @compute {
        csl.return
      }
      csl.export @y {alias = "y"}
      csl.export @compute {kind = "func"}
    }
    csl.layout {width = 1 : i64, height = 1 : i64} @layout {
      csl_layout.place @pe_b at (0, 0)
    }
    csl.host @main(%y: memref<32xf32>) {layout = @layout} {
      csl_host.launch @layout::@compute
      csl_host.memcpy_d2h @layout::@y to %y
          {px = 0 : i64, py = 0 : i64, width = 1 : i64, height = 1 : i64}
          : memref<32xf32>
    }
  }
}

// WA: const layout_mod = @import_module("<layout>");
// WA: var x: [16]f32;

// WB: const layout_mod = @import_module("<layout>");
// WB: var y: [32]f32;
