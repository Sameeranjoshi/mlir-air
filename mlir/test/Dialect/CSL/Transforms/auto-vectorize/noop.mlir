// RUN: air-opt %s -csl-auto-vectorize | FileCheck %s
//
// Smoke test: the pass registers and is a no-op on a CSL program that
// contains no scf.for loop.  We only verify it round-trips cleanly.

// CHECK-LABEL: csl.wafer @noop
module {
  csl.wafer @noop {arch = "wse3"} {
    csl.program @pe {
      csl.func @compute {
        csl.return
      }
      csl.export @compute {kind = "func"}
    }
    csl.layout {width = 1 : i64, height = 1 : i64} @layout {
      csl_layout.place @pe at (0, 0)
    }
    csl.host @main() {layout = @layout} {
      csl_host.launch @layout::@compute
    }
  }
}
