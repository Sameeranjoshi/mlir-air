// RUN: rm -rf %t && mkdir -p %t
// RUN: air-opt %s -csl-infer-exports | air-translate --emit-csl --output-dir=%t
// RUN: FileCheck --check-prefix=DOT    %s < %t/dot/pe.csl
// RUN: FileCheck --check-prefix=REDUCE %s < %t/reduce/pe.csl
// RUN: FileCheck --check-prefix=SAXPY  %s < %t/saxpy/pe.csl
// RUN: FileCheck --check-prefix=RELU   %s < %t/relu/pe.csl
//
// Task 10: exercise the arith op coverage added in task 2 via real-shape
// kernels. All wafers are 1-PE; the focus here is the function body.
//
// Kernels:
//   dot    — sum of elementwise product (addf + mulf)
//   reduce — sum a vector into a scalar (addf)
//   saxpy  — y[i] = a*x[i] + y[i] with a inlined as arith.constant (mulf + addf)
//   relu   — y[i] = max(x[i], 0.0) (maximumf)

// DOT-LABEL: fn compute() void
// DOT: while ({{i[0-9]+}} < 128)
// DOT: var {{t[0-9]+}}: f32 = {{t[0-9]+}} * {{t[0-9]+}};
// DOT: var {{t[0-9]+}}: f32 = {{t[0-9]+}} + {{t[0-9]+}};

// REDUCE-LABEL: fn compute() void
// REDUCE: while ({{i[0-9]+}} < 128)
// REDUCE: var {{t[0-9]+}}: f32 = {{t[0-9]+}} + {{t[0-9]+}};

// SAXPY-LABEL: fn compute() void
// SAXPY: while ({{i[0-9]+}} < 128)
// SAXPY: var {{t[0-9]+}}: f32 = 2.{{0+}} * {{t[0-9]+}};
// SAXPY: var {{t[0-9]+}}: f32 = {{t[0-9]+}} + {{t[0-9]+}};

// RELU-LABEL: fn compute() void
// RELU: while ({{i[0-9]+}} < 128)
// RELU: var {{t[0-9]+}}: f32 = if ({{t[0-9]+}} > 0.{{0+}}) {{t[0-9]+}} else 0.{{0+}};

module {
  // dot: out[0] = sum(a[i] * b[i]); we accumulate via load/store on out[0].
  csl.wafer @dot {arch = "wse3"} {
    csl.program @pe {
      %a = csl.var @a : memref<128xf32>
      %b = csl.var @b : memref<128xf32>
      %out = csl.var @out : memref<1xf32>
      csl.func @compute {
        %c0 = arith.constant 0 : index
        %n = arith.constant 128 : index
        %c1 = arith.constant 1 : index
        scf.for %i = %c0 to %n step %c1 {
          %va = memref.load %a[%i] : memref<128xf32>
          %vb = memref.load %b[%i] : memref<128xf32>
          %prod = arith.mulf %va, %vb : f32
          %acc = memref.load %out[%c0] : memref<1xf32>
          %sum = arith.addf %acc, %prod : f32
          memref.store %sum, %out[%c0] : memref<1xf32>
        }
        csl.return
      }
      csl.export @a {alias = "a"}
      csl.export @b {alias = "b"}
      csl.export @out {alias = "out"}
      csl.export @compute {kind = "func"}
    }
    csl.layout {width = 1 : i64, height = 1 : i64} @layout {
      csl_layout.place @pe at (0, 0)
    }
    csl.host @main(%a_in: memref<128xf32>, %b_in: memref<128xf32>,
                   %o_out: memref<1xf32>) {layout = @layout} {
      csl_host.memcpy_h2d %a_in to @layout::@a
          {px = 0 : i64, py = 0 : i64, width = 1 : i64, height = 1 : i64}
          : memref<128xf32>
      csl_host.memcpy_h2d %b_in to @layout::@b
          {px = 0 : i64, py = 0 : i64, width = 1 : i64, height = 1 : i64}
          : memref<128xf32>
      csl_host.launch @layout::@compute
      csl_host.memcpy_d2h @layout::@out to %o_out
          {px = 0 : i64, py = 0 : i64, width = 1 : i64, height = 1 : i64}
          : memref<1xf32>
    }
  }

  // reduce: out[0] = sum(a[i])
  csl.wafer @reduce {arch = "wse3"} {
    csl.program @pe {
      %a = csl.var @a : memref<128xf32>
      %out = csl.var @out : memref<1xf32>
      csl.func @compute {
        %c0 = arith.constant 0 : index
        %n = arith.constant 128 : index
        %c1 = arith.constant 1 : index
        scf.for %i = %c0 to %n step %c1 {
          %va = memref.load %a[%i] : memref<128xf32>
          %acc = memref.load %out[%c0] : memref<1xf32>
          %sum = arith.addf %acc, %va : f32
          memref.store %sum, %out[%c0] : memref<1xf32>
        }
        csl.return
      }
      csl.export @a {alias = "a"}
      csl.export @out {alias = "out"}
      csl.export @compute {kind = "func"}
    }
    csl.layout {width = 1 : i64, height = 1 : i64} @layout {
      csl_layout.place @pe at (0, 0)
    }
    csl.host @main(%a_in: memref<128xf32>, %o_out: memref<1xf32>)
        {layout = @layout} {
      csl_host.memcpy_h2d %a_in to @layout::@a
          {px = 0 : i64, py = 0 : i64, width = 1 : i64, height = 1 : i64}
          : memref<128xf32>
      csl_host.launch @layout::@compute
      csl_host.memcpy_d2h @layout::@out to %o_out
          {px = 0 : i64, py = 0 : i64, width = 1 : i64, height = 1 : i64}
          : memref<1xf32>
    }
  }

  // saxpy: y[i] = a*x[i] + y[i] with a = 2.0 (scalar param as arith.constant)
  csl.wafer @saxpy {arch = "wse3"} {
    csl.program @pe {
      %x = csl.var @x : memref<128xf32>
      %y = csl.var @y : memref<128xf32>
      csl.func @compute {
        %c0 = arith.constant 0 : index
        %n = arith.constant 128 : index
        %c1 = arith.constant 1 : index
        %a = arith.constant 2.0 : f32
        scf.for %i = %c0 to %n step %c1 {
          %vx = memref.load %x[%i] : memref<128xf32>
          %vy = memref.load %y[%i] : memref<128xf32>
          %scaled = arith.mulf %a, %vx : f32
          %r = arith.addf %scaled, %vy : f32
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
    csl.host @main(%x_in: memref<128xf32>, %y_io: memref<128xf32>)
        {layout = @layout} {
      csl_host.memcpy_h2d %x_in to @layout::@x
          {px = 0 : i64, py = 0 : i64, width = 1 : i64, height = 1 : i64}
          : memref<128xf32>
      csl_host.memcpy_h2d %y_io to @layout::@y
          {px = 0 : i64, py = 0 : i64, width = 1 : i64, height = 1 : i64}
          : memref<128xf32>
      csl_host.launch @layout::@compute
      csl_host.memcpy_d2h @layout::@y to %y_io
          {px = 0 : i64, py = 0 : i64, width = 1 : i64, height = 1 : i64}
          : memref<128xf32>
    }
  }

  // relu: y[i] = max(x[i], 0.0)
  csl.wafer @relu {arch = "wse3"} {
    csl.program @pe {
      %x = csl.var @x : memref<128xf32>
      %y = csl.var @y : memref<128xf32>
      csl.func @compute {
        %c0 = arith.constant 0 : index
        %n = arith.constant 128 : index
        %c1 = arith.constant 1 : index
        %zero = arith.constant 0.0 : f32
        scf.for %i = %c0 to %n step %c1 {
          %vx = memref.load %x[%i] : memref<128xf32>
          %r = arith.maximumf %vx, %zero : f32
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
}
