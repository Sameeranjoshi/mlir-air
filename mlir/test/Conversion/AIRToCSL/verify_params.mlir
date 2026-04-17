// RUN: air-opt %s -csl-verify-params -split-input-file -verify-diagnostics | FileCheck %s

// Valid: place binds the single declared param.
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

// Negative: extra param on place that program doesn't declare.
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

// Negative: missing param that program requires.
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

// Valid: no params on either side.
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

// -----

// Case (a): valid private helper call resolves.
// Unknown callees are caught by MLIR's built-in func.call verifier before
// -csl-verify-params runs (csl.program is a SymbolTable), so we only test
// the happy path here.  The pass still checks it defensively.
// CHECK-LABEL: csl.wafer @valid_call
module {
  csl.wafer @valid_call {arch = "wse3"} {
    csl.program @pe {
      func.func private @helper() { func.return }
      csl.func @compute {
        func.call @helper() : () -> ()
        csl.return
      }
    }
    csl.layout {width = 1 : i64, height = 1 : i64} @layout {
      csl_layout.place @pe at (0, 0)
    }
    csl.host @main() {layout = @layout} { }
  }
}

// -----

// Case (b): range-form params name is not a program block arg.
module {
  csl.wafer @bad_range_param {arch = "wse3"} {
    csl.program @pe(%pid: !csl.comptime<i16>) {
      csl.func @compute { csl.return }
    }
    csl.layout {width = 8 : i64, height = 1 : i64} @layout {
      // expected-error @+2 {{passes parameter 'foo' but @pe has no block arg 'foo'}}
      // expected-error @+1 {{missing parameter 'pid' required by @pe}}
      csl_layout.place @pe over [0:8, 0] vars (%i : i32) params {foo = %i : i16}
    }
    csl.host @main() {layout = @layout} { }
  }
}

// -----

// Case (b): valid range-form params name matches block arg.
// CHECK-LABEL: csl.wafer @good_range_param
module {
  csl.wafer @good_range_param {arch = "wse3"} {
    csl.program @pe(%pid: !csl.comptime<i16>) {
      csl.func @compute { csl.return }
    }
    csl.layout {width = 8 : i64, height = 1 : i64} @layout {
      csl_layout.place @pe over [0:8, 0] vars (%i : i32) params {pid = %i : i16}
    }
    csl.host @main() {layout = @layout} { }
  }
}

// -----

// Case (c): unequal sharding on h2d (255 elements, 8 PEs).
module {
  csl.wafer @unequal_h2d {arch = "wse3"} {
    csl.program @pe {
      %a = csl.var @a : memref<32xf32>
      csl.func @compute { csl.return }
      csl.export @a {alias = "a"}
      csl.export @compute {kind = "func"}
    }
    csl.layout {width = 8 : i64, height = 1 : i64} @layout {
      csl_layout.place @pe over [0:8, 0]
    }
    csl.host @main(%a_in: memref<255xf32>) {layout = @layout} {
      // expected-error @+1 {{memcpy_h2d buffer has 255 elements, not divisible by 8-PE placement}}
      csl_host.memcpy_h2d %a_in to @layout::@a
          {px = 0 : i64, py = 0 : i64, width = 8 : i64, height = 1 : i64}
          : memref<255xf32>
      csl_host.launch @layout::@compute
    }
  }
}

// -----

// Case (c): unequal sharding on d2h.
module {
  csl.wafer @unequal_d2h {arch = "wse3"} {
    csl.program @pe {
      %c = csl.var @c : memref<32xf32>
      csl.func @compute { csl.return }
      csl.export @c {alias = "c"}
      csl.export @compute {kind = "func"}
    }
    csl.layout {width = 8 : i64, height = 1 : i64} @layout {
      csl_layout.place @pe over [0:8, 0]
    }
    csl.host @main(%c_out: memref<255xf32>) {layout = @layout} {
      csl_host.launch @layout::@compute
      // expected-error @+1 {{memcpy_d2h buffer has 255 elements, not divisible by 8-PE placement}}
      csl_host.memcpy_d2h @layout::@c to %c_out
          {px = 0 : i64, py = 0 : i64, width = 8 : i64, height = 1 : i64}
          : memref<255xf32>
    }
  }
}

// -----

// Case (c): valid equal sharding (256 elements / 8 PEs = 32 each).
// CHECK-LABEL: csl.wafer @equal_h2d
module {
  csl.wafer @equal_h2d {arch = "wse3"} {
    csl.program @pe {
      %a = csl.var @a : memref<32xf32>
      csl.func @compute { csl.return }
      csl.export @a {alias = "a"}
      csl.export @compute {kind = "func"}
    }
    csl.layout {width = 8 : i64, height = 1 : i64} @layout {
      csl_layout.place @pe over [0:8, 0]
    }
    csl.host @main(%a_in: memref<256xf32>) {layout = @layout} {
      csl_host.memcpy_h2d %a_in to @layout::@a
          {px = 0 : i64, py = 0 : i64, width = 8 : i64, height = 1 : i64}
          : memref<256xf32>
      csl_host.launch @layout::@compute
    }
  }
}

// -----

// Case (d): non-private func.func at program level is rejected.
module {
  csl.wafer @non_private {arch = "wse3"} {
    csl.program @pe {
      // expected-error @+1 {{csl.program helpers must be 'private' func.func; @helper is not private}}
      func.func @helper() { func.return }
      csl.func @compute { csl.return }
    }
    csl.layout {width = 1 : i64, height = 1 : i64} @layout {
      csl_layout.place @pe at (0, 0)
    }
    csl.host @main() {layout = @layout} { }
  }
}

// -----

// Case (d): private func.func helper is accepted.
// CHECK-LABEL: csl.wafer @private_ok
module {
  csl.wafer @private_ok {arch = "wse3"} {
    csl.program @pe {
      func.func private @helper() { func.return }
      csl.func @compute {
        func.call @helper() : () -> ()
        csl.return
      }
    }
    csl.layout {width = 1 : i64, height = 1 : i64} @layout {
      csl_layout.place @pe at (0, 0)
    }
    csl.host @main() {layout = @layout} { }
  }
}
