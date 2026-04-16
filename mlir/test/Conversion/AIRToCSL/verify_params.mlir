// RUN: air-opt %s -csl-verify-params -split-input-file -verify-diagnostics | FileCheck %s

// ---- Valid: place binds the single declared param ----

// CHECK-LABEL: csl.wafer @valid
module {
  csl.wafer @valid {arch = "wse3"} {
    csl.program @pe(%M: !csl.comptime<i16>) { }
    csl.layout {width = 1 : i64, height = 1 : i64} @layout {
      csl_layout.place @pe at (0, 0) {M = 4 : i16}
    }
    csl.host @main() {layout = @layout} { }
  }
}

// -----

// ---- Negative: extra param on place that program doesn't declare ----

module {
  csl.wafer @extra_param {arch = "wse3"} {
    csl.program @pe(%M: !csl.comptime<i16>) { }
    csl.layout {width = 1 : i64, height = 1 : i64} @layout {
      // expected-error @+1 {{passes parameter 'N' but @pe has no matching block argument}}
      csl_layout.place @pe at (0, 0) {M = 4 : i16, N = 6 : i16}
    }
    csl.host @main() {layout = @layout} { }
  }
}

// -----

// ---- Negative: missing param that program requires ----

module {
  csl.wafer @missing_param {arch = "wse3"} {
    csl.program @pe(%M: !csl.comptime<i16>, %N: !csl.comptime<i16>) { }
    csl.layout {width = 1 : i64, height = 1 : i64} @layout {
      // expected-error @+1 {{missing parameter 'N' required by @pe}}
      csl_layout.place @pe at (0, 0) {M = 4 : i16}
    }
    csl.host @main() {layout = @layout} { }
  }
}

// -----

// ---- Valid: no params on either side ----

// CHECK-LABEL: csl.wafer @no_params
module {
  csl.wafer @no_params {arch = "wse3"} {
    csl.program @pe { }
    csl.layout {width = 1 : i64, height = 1 : i64} @layout {
      csl_layout.place @pe at (0, 0)
    }
    csl.host @main() {layout = @layout} { }
  }
}
