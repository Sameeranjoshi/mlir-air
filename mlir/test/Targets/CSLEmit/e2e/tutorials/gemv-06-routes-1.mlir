// REQUIRES: cerebras-sdk
//
// RUN: rm -rf %t && mkdir -p %t
// RUN: air-opt --csl-dataflow-to-csl %s | air-translate --emit-csl --output-dir=%t
// RUN: cd %t/gemv06 && bash commands_wse3.sh 2>&1 | FileCheck %s
//
// Tutorial 06 — 2-PE GEMV with inter-PE routing (routes-1).
// Reproduces SDK tutorial gemv-06-routes-1:
//   y = A * x  (no bias),  N=6 split 3 columns per PE.
//
// Key CSL concepts:
//   - Two distinct programs for left PE (sender) and right PE (receiver)
//   - Left PE computes partial y from its A/x shard, sends via fabric channel
//   - Right PE computes partial y from its A/x shard, receives the left's
//     partial, adds them together to produce the final y
//   - csl.dataflow.put / csl.dataflow.get: high-level inter-PE dataflow ops
//   - --csl-dataflow-to-csl lowers to colors, DSDs, tasks, queue init
//   - csl_layout.dataflow @send_ch from(0,0) to(1,0): declares the channel
//
// Left PE (x=0): computes y_L = A_L * x_L, sends y_L east via @send_ch
// Right PE (x=1): computes y_R = A_R * x_R, receives y_L, computes y = y_L + y_R
//
// The right PE's final y buffer is read back by the host.
//
// Corresponding SDK tutorial: gemv-06-routes-1/pe_program.csl

// CHECK: SUCCESS!

csl.wafer @gemv06 {arch = "wse3"} {
  // Left PE: computes partial y from left half of A and x, sends east
  csl.program @left_pe {
    %A = csl.var @A : memref<12xf32>  // M=4, N_per_PE=3 column-major shard
    %x = csl.var @x : memref<3xf32>
    %y = csl.var @y : memref<4xf32>  // partial result; sent via fabric

    csl.func @compute {
      %c0  = arith.constant 0 : index
      %c1  = arith.constant 1 : index
      %m   = arith.constant 4 : index  // M = 4
      %npp = arith.constant 3 : index  // N_per_PE = 3

      // Initialize y to zero (it will accumulate the partial sum)
      %zero = arith.constant 0.0 : f32
      scf.for %i = %c0 to %m step %c1 {
        memref.store %zero, %y[%i] : memref<4xf32>
      }

      // Partial GEMV: y += A[:,j] * x[j] for j in 0..N_per_PE
      %y_dsd = csl.get_mem_dsd %y : memref<4xf32> -> !csl.dsd
      scf.for %j = %c0 to %npp step %c1 {
        %col_view = memref.subview %A[%j] [4] [3]
                    : memref<12xf32> to memref<4xf32, strided<[3], offset: ?>>
        %A_col_dsd = csl.get_mem_dsd %col_view
                     : memref<4xf32, strided<[3], offset: ?>> -> !csl.dsd
        %xj = memref.load %x[%j] : memref<3xf32>
        csl.builtin_call "fmacs"(%y_dsd, %y_dsd, %A_col_dsd, %xj)
            : (!csl.dsd, !csl.dsd, !csl.dsd, f32) -> ()
      }
      // Send partial y to the right PE via fabric channel @send_ch
      %extent = arith.constant 4 : index
      csl.dataflow.put @send_ch source(%y) extent(%extent : index) : memref<4xf32>
      csl.return
    }
    csl.export @A {alias = "A_left",  direction = "in"}
    csl.export @x {alias = "x_left",  direction = "in"}
    csl.export @compute {kind = "func", direction = "internal"}
  }

  // Right PE: computes its own partial y, receives left's partial, sums them
  csl.program @right_pe {
    %A = csl.var @A : memref<12xf32>  // Right half of A (columns 3..5)
    %x = csl.var @x : memref<3xf32>
    %y = csl.var @y : memref<4xf32>  // Final output (left partial + right partial)

    csl.func @compute {
      %c0  = arith.constant 0 : index
      %c1  = arith.constant 1 : index
      %m   = arith.constant 4 : index
      %npp = arith.constant 3 : index

      // Initialize y to zero
      %zero = arith.constant 0.0 : f32
      scf.for %i = %c0 to %m step %c1 {
        memref.store %zero, %y[%i] : memref<4xf32>
      }

      // Partial GEMV for right half: y += A[:,j] * x[j]
      %y_dsd = csl.get_mem_dsd %y : memref<4xf32> -> !csl.dsd
      scf.for %j = %c0 to %npp step %c1 {
        %col_view = memref.subview %A[%j] [4] [3]
                    : memref<12xf32> to memref<4xf32, strided<[3], offset: ?>>
        %A_col_dsd = csl.get_mem_dsd %col_view
                     : memref<4xf32, strided<[3], offset: ?>> -> !csl.dsd
        %xj = memref.load %x[%j] : memref<3xf32>
        csl.builtin_call "fmacs"(%y_dsd, %y_dsd, %A_col_dsd, %xj)
            : (!csl.dsd, !csl.dsd, !csl.dsd, f32) -> ()
      }
      // Receive left PE's partial y and accumulate: y += y_left (via @fadds)
      %extent = arith.constant 4 : index
      csl.dataflow.get @send_ch target(%y) extent(%extent : index) : memref<4xf32>
      csl.return
    }
    csl.export @A {alias = "A_right", direction = "in"}
    csl.export @x {alias = "x_right", direction = "in"}
    csl.export @y {alias = "y_out",   direction = "out"}
    csl.export @compute {kind = "func", direction = "internal"}
  }

  csl.layout {width = 2 : i64, height = 1 : i64} @layout {
    // Declare fabric channel: left_pe sends to right_pe (EAST direction)
    csl_layout.dataflow @send_ch from(0, 0) to(1, 0)
    csl_layout.place  @left_pe  at (0, 0)
    csl_layout.place  @right_pe at (1, 0)
    csl_layout.export "A_left"  from @left_pe::@A
    csl_layout.export "x_left"  from @left_pe::@x
    csl_layout.export "A_right" from @right_pe::@A
    csl_layout.export "x_right" from @right_pe::@x
    csl_layout.export "y_out"   from @right_pe::@y
    csl_layout.export "compute" from @left_pe::@compute {kind = "func"}
  }

  csl.host @main(
      %A_left_in:  memref<12xf32>,
      %x_left_in:  memref<3xf32>,
      %A_right_in: memref<12xf32>,
      %x_right_in: memref<3xf32>,
      %y_out:      memref<4xf32>) {layout = @layout} {
    csl_host.memcpy_h2d %A_left_in to @layout::@A_left
        {px = 0 : i64, py = 0 : i64, width = 1 : i64, height = 1 : i64}
        : memref<12xf32>
    csl_host.memcpy_h2d %x_left_in to @layout::@x_left
        {px = 0 : i64, py = 0 : i64, width = 1 : i64, height = 1 : i64}
        : memref<3xf32>
    csl_host.memcpy_h2d %A_right_in to @layout::@A_right
        {px = 1 : i64, py = 0 : i64, width = 1 : i64, height = 1 : i64}
        : memref<12xf32>
    csl_host.memcpy_h2d %x_right_in to @layout::@x_right
        {px = 1 : i64, py = 0 : i64, width = 1 : i64, height = 1 : i64}
        : memref<3xf32>
    csl_host.launch @layout::@compute
    csl_host.memcpy_d2h @layout::@y_out to %y_out
        {px = 1 : i64, py = 0 : i64, width = 1 : i64, height = 1 : i64}
        : memref<4xf32>
  }
}
