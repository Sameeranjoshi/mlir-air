// RUN: rm -rf %t && mkdir -p %t
// RUN: air-opt %s -csl-infer-exports | air-translate --emit-csl --output-dir=%t
// RUN: FileCheck --check-prefix=DIV %s < %t/div_const/pe.csl
// RUN: FileCheck --check-prefix=MIN %s < %t/min_clamp/pe.csl
// RUN: FileCheck --check-prefix=NEG %s < %t/neg_signflip/pe.csl
// RUN: FileCheck --check-prefix=XOR %s < %t/xor_disagree/pe.csl
//
// Coverage for arith ops handled by the emitter but not exercised by any
// existing e2e wafer: arith.divf, arith.minimumf, arith.negf.
// Plus a regression test for the i1 XOR fix (commit 32b125e0 follow-up):
// arith.xori on i1 must emit `a != b` (logical xor on bool), not bitwise `^`.

// DIV-LABEL: fn compute() void
// DIV: while (
// DIV: var {{t[0-9]+}}: f32 = {{t[0-9]+}} / 2.{{0+}};

// MIN-LABEL: fn compute() void
// MIN: while (
// MIN: var {{t[0-9]+}}: f32 = if ({{t[0-9]+}} < 0.5{{0*}}) {{t[0-9]+}} else 0.5{{0*}};

// NEG-LABEL: fn compute() void
// NEG: while (
// NEG: var {{t[0-9]+}}: f32 = -{{t[0-9]+}};

// XOR-LABEL: fn compute() void
// XOR: while (
// XOR: var {{t[0-9]+}}: bool = {{b[0-9]+}} != {{b[0-9]+}};

module {
  // div_const: y[i] = x[i] / 2.0   (arith.divf)
  csl.wafer @div_const {arch = "wse3"} {
    csl.program @pe {
      %x = csl.var @x : memref<128xf32>
      %y = csl.var @y : memref<128xf32>
      csl.func @compute {
        %c0 = arith.constant 0 : index
        %n  = arith.constant 128 : index
        %c1 = arith.constant 1 : index
        %two = arith.constant 2.0 : f32
        scf.for %i = %c0 to %n step %c1 {
          %vx = memref.load %x[%i] : memref<128xf32>
          %r = arith.divf %vx, %two : f32
          memref.store %r, %y[%i] : memref<128xf32>
        }
        csl.return
      }
      csl.export @x {alias = "x"}
      csl.export @y {alias = "y"}
      csl.export @compute {kind = "func"}
    }
    csl.layout {width = 1 : i64, height = 1 : i64} @layout {
      csl_layout.place @pe at (0, 0)
    }
    csl.host @main(%x_in: memref<128xf32>, %y_out: memref<128xf32>)
        {layout = @layout} {
      csl_host.memcpy_h2d %x_in to @layout::@x
          {px = 0 : i64, py = 0 : i64, width = 1 : i64, height = 1 : i64}
          : memref<128xf32>
      csl_host.launch @layout::@compute
      csl_host.memcpy_d2h @layout::@y to %y_out
          {px = 0 : i64, py = 0 : i64, width = 1 : i64, height = 1 : i64}
          : memref<128xf32>
    }
  }

  // min_clamp: y[i] = min(x[i], 0.5)   (arith.minimumf)
  csl.wafer @min_clamp {arch = "wse3"} {
    csl.program @pe {
      %x = csl.var @x : memref<128xf32>
      %y = csl.var @y : memref<128xf32>
      csl.func @compute {
        %c0 = arith.constant 0 : index
        %n  = arith.constant 128 : index
        %c1 = arith.constant 1 : index
        %half = arith.constant 0.5 : f32
        scf.for %i = %c0 to %n step %c1 {
          %vx = memref.load %x[%i] : memref<128xf32>
          %r = arith.minimumf %vx, %half : f32
          memref.store %r, %y[%i] : memref<128xf32>
        }
        csl.return
      }
      csl.export @x {alias = "x"}
      csl.export @y {alias = "y"}
      csl.export @compute {kind = "func"}
    }
    csl.layout {width = 1 : i64, height = 1 : i64} @layout {
      csl_layout.place @pe at (0, 0)
    }
    csl.host @main(%x_in: memref<128xf32>, %y_out: memref<128xf32>)
        {layout = @layout} {
      csl_host.memcpy_h2d %x_in to @layout::@x
          {px = 0 : i64, py = 0 : i64, width = 1 : i64, height = 1 : i64}
          : memref<128xf32>
      csl_host.launch @layout::@compute
      csl_host.memcpy_d2h @layout::@y to %y_out
          {px = 0 : i64, py = 0 : i64, width = 1 : i64, height = 1 : i64}
          : memref<128xf32>
    }
  }

  // neg_signflip: y[i] = -x[i]   (arith.negf)
  csl.wafer @neg_signflip {arch = "wse3"} {
    csl.program @pe {
      %x = csl.var @x : memref<128xf32>
      %y = csl.var @y : memref<128xf32>
      csl.func @compute {
        %c0 = arith.constant 0 : index
        %n  = arith.constant 128 : index
        %c1 = arith.constant 1 : index
        scf.for %i = %c0 to %n step %c1 {
          %vx = memref.load %x[%i] : memref<128xf32>
          %r = arith.negf %vx : f32
          memref.store %r, %y[%i] : memref<128xf32>
        }
        csl.return
      }
      csl.export @x {alias = "x"}
      csl.export @y {alias = "y"}
      csl.export @compute {kind = "func"}
    }
    csl.layout {width = 1 : i64, height = 1 : i64} @layout {
      csl_layout.place @pe at (0, 0)
    }
    csl.host @main(%x_in: memref<128xf32>, %y_out: memref<128xf32>)
        {layout = @layout} {
      csl_host.memcpy_h2d %x_in to @layout::@x
          {px = 0 : i64, py = 0 : i64, width = 1 : i64, height = 1 : i64}
          : memref<128xf32>
      csl_host.launch @layout::@compute
      csl_host.memcpy_d2h @layout::@y to %y_out
          {px = 0 : i64, py = 0 : i64, width = 1 : i64, height = 1 : i64}
          : memref<128xf32>
    }
  }

  // xor_disagree: y[i] = 1.0 if (i < N/2) XOR (x[i] > 0.5) else 0.0
  // Regression for commit 32b125e0 follow-up: XOrIOp on i1 must emit `!=`
  // (logical xor) — bitwise `^` is rejected by CSL on bool operands.
  csl.wafer @xor_disagree {arch = "wse3"} {
    csl.program @pe {
      %x = csl.var @x : memref<128xf32>
      %y = csl.var @y : memref<128xf32>
      csl.func @compute {
        %c0     = arith.constant 0   : index
        %cHalfN = arith.constant 64  : index
        %cN     = arith.constant 128 : index
        %c1     = arith.constant 1   : index
        %thr    = arith.constant 0.5 : f32
        %one    = arith.constant 1.0 : f32
        %zero   = arith.constant 0.0 : f32
        scf.for %i = %c0 to %cN step %c1 {
          %first_half = arith.cmpi slt, %i, %cHalfN : index
          %vx         = memref.load %x[%i] : memref<128xf32>
          %above      = arith.cmpf ogt, %vx, %thr : f32
          %disagree   = arith.xori %first_half, %above : i1
          memref.store %zero, %y[%i] : memref<128xf32>
          scf.if %disagree {
            memref.store %one, %y[%i] : memref<128xf32>
          }
        }
        csl.return
      }
      csl.export @x {alias = "x"}
      csl.export @y {alias = "y"}
      csl.export @compute {kind = "func"}
    }
    csl.layout {width = 1 : i64, height = 1 : i64} @layout {
      csl_layout.place @pe at (0, 0)
    }
    csl.host @main(%x_in: memref<128xf32>, %y_out: memref<128xf32>)
        {layout = @layout} {
      csl_host.memcpy_h2d %x_in to @layout::@x
          {px = 0 : i64, py = 0 : i64, width = 1 : i64, height = 1 : i64}
          : memref<128xf32>
      csl_host.launch @layout::@compute
      csl_host.memcpy_d2h @layout::@y to %y_out
          {px = 0 : i64, py = 0 : i64, width = 1 : i64, height = 1 : i64}
          : memref<128xf32>
    }
  }
}
