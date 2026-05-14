// RUN: rm -rf %t && mkdir -p %t
// RUN: air-opt %s -csl-infer-exports | air-translate --emit-csl --output-dir=%t
// RUN: FileCheck %s --check-prefix=PE     < %t/simd/pe.csl
// RUN: FileCheck %s --check-prefix=LAYOUT < %t/simd/layout.csl
// RUN: FileCheck %s --check-prefix=HOST    < %t/simd/run.py
//
// v5 Task 8 — N-PE SIMD e2e.
//
// Proves that v4's subgrid placement (csl_layout.place @pe over [0:W, 0:H])
// composes with v5's DSD compute with NO emitter code change. Same program
// runs on every PE; host-visible memrefs are equal-sharded across the 4x2
// subgrid (8 PEs, 256 total elements → 32 elements per PE).
//
// Alpha is inlined as a kernel-scope arith.constant — broadcast style, no
// host-side broadcast memcpy needed.

// PE-LABEL: fn compute() void
// PE: const {{.*}} = @get_dsd(mem1d_dsd, .{ .base_address = &A, .extent = 32 });
// PE: const {{.*}} = @get_dsd(mem1d_dsd, .{ .base_address = &y, .extent = 32 });
// PE: @fmacs(

// LAYOUT: @set_rectangle(4, 2);
// LAYOUT: while (

// HOST: runner.memcpy_h2d(runner.get_id("A"), arg0, 0, 0, 4, 2, 32,
// HOST: runner.memcpy_d2h(arg1, runner.get_id("y"), 0, 0, 4, 2, 32,

module {
  csl.wafer @simd {arch = "wse3"} {
    csl.program @pe {
      // Per-PE shards: each of the 8 PEs owns a 32-elem slice of A and y.
      %A = csl.var @A : memref<32xf32>
      %y = csl.var @y : memref<32xf32>
      csl.func @compute {
        %n = arith.constant 32  : index
        %a = arith.constant 2.0 : f32
        %Ad = csl.get_mem_dsd %A : memref<32xf32> -> !csl.dsd
        %yd = csl.get_mem_dsd %y : memref<32xf32> -> !csl.dsd
        csl.builtin_call "fmacs"(%yd, %yd, %Ad, %a)
            : (!csl.dsd, !csl.dsd, !csl.dsd, f32) -> ()
        csl.return
      }
      csl.export @A {alias = "A"}
      csl.export @y {alias = "y"}
      csl.export @compute {kind = "func"}
    }
    // 4 x 2 subgrid placement via v4 `over` form — each PE runs the same
    // `@pe` program with its own 32-elem shards.
    csl.layout {width = 4 : i64, height = 2 : i64} @layout {
      csl_layout.place @pe over [0:4, 0:2]
    }
    csl.host @main(%A_in: memref<8x32xf32>, %y_io: memref<8x32xf32>)
        {layout = @layout} {
      csl_host.memcpy_h2d %A_in to @layout::@A
          {px = 0 : i64, py = 0 : i64, width = 4 : i64, height = 2 : i64}
          : memref<8x32xf32>
      csl_host.memcpy_h2d %y_io to @layout::@y
          {px = 0 : i64, py = 0 : i64, width = 4 : i64, height = 2 : i64}
          : memref<8x32xf32>
      csl_host.launch @layout::@compute
      csl_host.memcpy_d2h @layout::@y to %y_io
          {px = 0 : i64, py = 0 : i64, width = 4 : i64, height = 2 : i64}
          : memref<8x32xf32>
    }
  }
}
