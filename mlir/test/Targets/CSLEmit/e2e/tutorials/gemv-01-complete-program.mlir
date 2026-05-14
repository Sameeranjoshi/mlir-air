// RUN: rm -rf %t && mkdir -p %t
// RUN: air-opt %s -csl-infer-exports | air-translate --emit-csl --output-dir=%t
// RUN: FileCheck %s < %t/gemv01/pe.csl
//
// Tutorial 01 — GEMV complete program on a single PE.
// Reproduces SDK tutorial gemv-01-complete-program:
//   y[i] = sum_j A[i*N + j] * x[j] + b[i]   (M=4, N=6)
//
// The PE self-initializes all arrays (no host data transfer for inputs).
// The host only reads back the result via memcpy_d2h.
//
// Key CSL concepts introduced:
//   - Module-level var declarations (A, x, b, y)
//   - Nested for loops for scalar GEMV
//   - Export of result pointer + entry function
//
// Corresponding SDK tutorial: gemv-01-complete-program/pe_program.csl

// CHECK-LABEL: fn init_and_compute() void
// CHECK: while (
// CHECK: while (
// CHECK: * {{[a-z_0-9]+}};
// CHECK: y[

module {
  csl.wafer @gemv01 {arch = "wse3"} {
    csl.program @pe {
      %A = csl.var @A : memref<24xf32>  // [M*N] = [4*6] row-major
      %x = csl.var @x : memref<6xf32>
      %b = csl.var @b : memref<4xf32>
      %y = csl.var @y : memref<4xf32>

      csl.func @init_and_compute {
        %c0  = arith.constant 0 : index
        %c1  = arith.constant 1 : index
        %m   = arith.constant 4 : index   // M = 4 rows
        %n   = arith.constant 6 : index   // N = 6 cols
        %mn  = arith.constant 24 : index  // M*N = 24 elements

        // Initialize A[idx] = 1.0 for all elements
        // (SDK tutorial uses (float)idx, but arith.sitofp is not yet supported
        // by the CSL emitter; constant initialization demonstrates the pattern)
        %one = arith.constant 1.0 : f32
        scf.for %idx = %c0 to %mn step %c1 {
          memref.store %one, %A[%idx] : memref<24xf32>
        }
        // Initialize x[j] = 2.0, b[i] = 3.0, y[i] = 0.0
        // (values chosen so y[i] = sum_j A[i,j]*x[j] + b[i] = 6*1.0*2.0 + 3.0 = 15)
        %two  = arith.constant 2.0 : f32
        %three = arith.constant 3.0 : f32
        %zero = arith.constant 0.0 : f32
        scf.for %j = %c0 to %n step %c1 {
          memref.store %two, %x[%j] : memref<6xf32>
        }
        scf.for %i = %c0 to %m step %c1 {
          memref.store %three, %b[%i] : memref<4xf32>
          memref.store %zero,  %y[%i] : memref<4xf32>
        }
        // Compute GEMV: y[i] += A[i*N + j] * x[j], then y[i] += b[i]
        scf.for %i = %c0 to %m step %c1 {
          scf.for %j = %c0 to %n step %c1 {
            %iN   = arith.muli %i, %n    : index
            %iNj  = arith.addi %iN, %j   : index
            %aij  = memref.load %A[%iNj] : memref<24xf32>
            %xj   = memref.load %x[%j]   : memref<6xf32>
            %prod = arith.mulf %aij, %xj  : f32
            %yi   = memref.load %y[%i]   : memref<4xf32>
            %sum  = arith.addf %yi, %prod : f32
            memref.store %sum, %y[%i]    : memref<4xf32>
          }
          // y[i] += b[i]
          %bi  = memref.load %b[%i] : memref<4xf32>
          %yi2 = memref.load %y[%i] : memref<4xf32>
          %yb  = arith.addf %yi2, %bi : f32
          memref.store %yb, %y[%i]    : memref<4xf32>
        }
        csl.return
      }
      csl.export @y {alias = "y"}
      csl.export @init_and_compute {kind = "func"}
    }
    csl.layout {width = 1 : i64, height = 1 : i64} @layout {
      csl_layout.place @pe at (0, 0)
    }
    // Host only reads y — the PE self-initializes A, x, b.
    csl.host @main(%y_out: memref<4xf32>) {layout = @layout} {
      csl_host.launch @layout::@init_and_compute
      csl_host.memcpy_d2h @layout::@y to %y_out
          {px = 0 : i64, py = 0 : i64, width = 1 : i64, height = 1 : i64}
          : memref<4xf32>
    }
  }
}
