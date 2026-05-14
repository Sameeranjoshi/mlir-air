// RUN: rm -rf %t && mkdir -p %t
// RUN: air-opt %s -csl-infer-exports | air-translate --emit-csl --output-dir=%t
// RUN: FileCheck %s --check-prefix=CLAMP < %t/clamp_neg/pe.csl
// RUN: FileCheck %s --check-prefix=BOUND < %t/bound_i/pe.csl
//
// v5 Task 5 — emit scf.if + arith.cmpf / arith.cmpi in kernel scope.
// No yielded values (that form is out of v5 scope).

// CLAMP-LABEL: fn compute() void
// CLAMP: while (
// CLAMP: const {{.*}} = {{.*}} < 0.{{0+}};
// CLAMP: if (
// CLAMP: x[{{.*}}] = 0.
// CLAMP: }

// BOUND-LABEL: fn compute() void
// BOUND: while (
// BOUND: const {{.*}} = {{.*}} == 0;
// BOUND: if (
// BOUND: }
// BOUND: else

module {
  // Clamp-negative-to-zero: demonstrate scf.if + arith.cmpf (olt).
  csl.wafer @clamp_neg {arch = "wse3"} {
    csl.program @pe {
      %x = csl.var @x : memref<8xf32>
      csl.func @compute {
        %c0   = arith.constant 0   : index
        %c1   = arith.constant 1   : index
        %c8   = arith.constant 8   : index
        %zero = arith.constant 0.0 : f32
        scf.for %i = %c0 to %c8 step %c1 {
          %v  = memref.load %x[%i] : memref<8xf32>
          %lt = arith.cmpf olt, %v, %zero : f32
          scf.if %lt {
            memref.store %zero, %x[%i] : memref<8xf32>
          }
        }
        csl.return
      }
      csl.export @x {alias = "x"}
      csl.export @compute {kind = "func"}
    }
    csl.layout {width = 1 : i64, height = 1 : i64} @layout {
      csl_layout.place @pe at (0, 0)
    }
    csl.host @main(%x_io: memref<8xf32>) {layout = @layout} {
      csl_host.memcpy_h2d %x_io to @layout::@x
          {px = 0 : i64, py = 0 : i64, width = 1 : i64, height = 1 : i64}
          : memref<8xf32>
      csl_host.launch @layout::@compute
      csl_host.memcpy_d2h @layout::@x to %x_io
          {px = 0 : i64, py = 0 : i64, width = 1 : i64, height = 1 : i64}
          : memref<8xf32>
    }
  }

  // Endpoint vs interior: demonstrate scf.if/else + arith.cmpi.
  csl.wafer @bound_i {arch = "wse3"} {
    csl.program @pe {
      %x = csl.var @x : memref<8xf32>
      %y = csl.var @y : memref<8xf32>
      csl.func @compute {
        %c0   = arith.constant 0   : index
        %c1   = arith.constant 1   : index
        %c8   = arith.constant 8   : index
        %i0i  = arith.constant 0   : i32
        %zero = arith.constant 0.0 : f32
        %two  = arith.constant 2.0 : f32
        scf.for %i = %c0 to %c8 step %c1 {
          %vi = arith.index_cast %i : index to i32
          %is_zero = arith.cmpi eq, %vi, %i0i : i32
          scf.if %is_zero {
            // endpoint: copy input
            %v = memref.load %x[%i] : memref<8xf32>
            memref.store %v, %y[%i] : memref<8xf32>
          } else {
            // interior: double
            %v = memref.load %x[%i] : memref<8xf32>
            %d = arith.mulf %v, %two : f32
            memref.store %d, %y[%i] : memref<8xf32>
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
    csl.host @main(%x_in: memref<8xf32>, %y_out: memref<8xf32>)
        {layout = @layout} {
      csl_host.memcpy_h2d %x_in to @layout::@x
          {px = 0 : i64, py = 0 : i64, width = 1 : i64, height = 1 : i64}
          : memref<8xf32>
      csl_host.launch @layout::@compute
      csl_host.memcpy_d2h @layout::@y to %y_out
          {px = 0 : i64, py = 0 : i64, width = 1 : i64, height = 1 : i64}
          : memref<8xf32>
    }
  }
}
