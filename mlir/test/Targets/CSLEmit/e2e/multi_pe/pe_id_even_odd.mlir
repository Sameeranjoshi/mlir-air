// REQUIRES: cerebras-sdk
//
// RUN: rm -rf %t && mkdir -p %t
// RUN: air-opt --csl-dataflow-to-csl %s | air-translate --emit-csl --output-dir=%t
// RUN: cd %t/pe_id_even_odd && bash commands_wse3.sh 2>&1 | FileCheck %s
//
// 4 independent PEs on a 1x4 grid with no inter-PE fabric communication.
// Each PE applies a local per-PE transform to its input buffer and writes the
// result to its output buffer:
//
//   pe0 (even, x=0): out[i] = in[i] - 1.0
//   pe1 (odd,  x=1): out[i] = in[i] + 1.0
//   pe2 (even, x=2): out[i] = in[i] - 1.0
//   pe3 (odd,  x=3): out[i] = in[i] + 1.0
//
// There are no csl_layout.dataflow ops so no fabric DSDs are emitted.
// hasFabricOp = false in the host emitter → the passthrough equality heuristic
// does NOT fire; the emitter falls back to a NaN/inf sanity check on all four
// output buffers.  SUCCESS! indicates all outputs are finite numbers.
//
// This test exercises:
//   1. Multiple distinct programs with different compute bodies
//   2. Two-buffer-per-PE pattern (separate in/out vars) with in-place loop
//   3. scf.for + memref.load/store + arith.addf/-1.0 vs +1.0 per PE type
//   4. Four PEs running concurrently with independent compute
//
// CHECK: SUCCESS!

csl.wafer @pe_id_even_odd {arch = "wse3"} {
  // Even PE: out[i] = in[i] - 1.0
  csl.program @pe0 {
    %buf_in  = csl.var @buf_in  : memref<64xf32>
    %buf_out = csl.var @buf_out : memref<64xf32>
    csl.func @compute {
      %c0   = arith.constant 0    : index
      %n    = arith.constant 64   : index
      %c1   = arith.constant 1    : index
      %neg1 = arith.constant -1.0 : f32
      scf.for %i = %c0 to %n step %c1 {
        %v   = memref.load  %buf_in[%i]  : memref<64xf32>
        %nv  = arith.addf %v, %neg1      : f32
        memref.store %nv, %buf_out[%i]   : memref<64xf32>
      }
      csl.return
    }
    csl.export @buf_in  {alias = "buf_pe0_in",  direction = "in"}
    csl.export @buf_out {alias = "buf_pe0_out", direction = "out"}
    csl.export @compute {kind = "func", direction = "internal"}
  }
  // Odd PE: out[i] = in[i] + 1.0
  csl.program @pe1 {
    %buf_in  = csl.var @buf_in  : memref<64xf32>
    %buf_out = csl.var @buf_out : memref<64xf32>
    csl.func @compute {
      %c0  = arith.constant 0   : index
      %n   = arith.constant 64  : index
      %c1  = arith.constant 1   : index
      %pos1 = arith.constant 1.0 : f32
      scf.for %i = %c0 to %n step %c1 {
        %v   = memref.load  %buf_in[%i]  : memref<64xf32>
        %nv  = arith.addf %v, %pos1      : f32
        memref.store %nv, %buf_out[%i]   : memref<64xf32>
      }
      csl.return
    }
    csl.export @buf_in  {alias = "buf_pe1_in",  direction = "in"}
    csl.export @buf_out {alias = "buf_pe1_out", direction = "out"}
    csl.export @compute {kind = "func", direction = "internal"}
  }
  // Even PE: out[i] = in[i] - 1.0
  csl.program @pe2 {
    %buf_in  = csl.var @buf_in  : memref<64xf32>
    %buf_out = csl.var @buf_out : memref<64xf32>
    csl.func @compute {
      %c0   = arith.constant 0    : index
      %n    = arith.constant 64   : index
      %c1   = arith.constant 1    : index
      %neg1 = arith.constant -1.0 : f32
      scf.for %i = %c0 to %n step %c1 {
        %v   = memref.load  %buf_in[%i]  : memref<64xf32>
        %nv  = arith.addf %v, %neg1      : f32
        memref.store %nv, %buf_out[%i]   : memref<64xf32>
      }
      csl.return
    }
    csl.export @buf_in  {alias = "buf_pe2_in",  direction = "in"}
    csl.export @buf_out {alias = "buf_pe2_out", direction = "out"}
    csl.export @compute {kind = "func", direction = "internal"}
  }
  // Odd PE: out[i] = in[i] + 1.0
  csl.program @pe3 {
    %buf_in  = csl.var @buf_in  : memref<64xf32>
    %buf_out = csl.var @buf_out : memref<64xf32>
    csl.func @compute {
      %c0  = arith.constant 0   : index
      %n   = arith.constant 64  : index
      %c1  = arith.constant 1   : index
      %pos1 = arith.constant 1.0 : f32
      scf.for %i = %c0 to %n step %c1 {
        %v   = memref.load  %buf_in[%i]  : memref<64xf32>
        %nv  = arith.addf %v, %pos1      : f32
        memref.store %nv, %buf_out[%i]   : memref<64xf32>
      }
      csl.return
    }
    csl.export @buf_in  {alias = "buf_pe3_in",  direction = "in"}
    csl.export @buf_out {alias = "buf_pe3_out", direction = "out"}
    csl.export @compute {kind = "func", direction = "internal"}
  }
  csl.layout {width = 4 : i64, height = 1 : i64} @layout {
    csl_layout.place  @pe0 at (0, 0)
    csl_layout.place  @pe1 at (1, 0)
    csl_layout.place  @pe2 at (2, 0)
    csl_layout.place  @pe3 at (3, 0)
    csl_layout.export "buf_pe0_in"  from @pe0::@buf_in
    csl_layout.export "buf_pe0_out" from @pe0::@buf_out
    csl_layout.export "buf_pe1_in"  from @pe1::@buf_in
    csl_layout.export "buf_pe1_out" from @pe1::@buf_out
    csl_layout.export "buf_pe2_in"  from @pe2::@buf_in
    csl_layout.export "buf_pe2_out" from @pe2::@buf_out
    csl_layout.export "buf_pe3_in"  from @pe3::@buf_in
    csl_layout.export "buf_pe3_out" from @pe3::@buf_out
    csl_layout.export "compute"     from @pe0::@compute  {kind = "func"}
  }
  csl.host @main(
      %in0:  memref<64xf32>, %out0: memref<64xf32>,
      %in1:  memref<64xf32>, %out1: memref<64xf32>,
      %in2:  memref<64xf32>, %out2: memref<64xf32>,
      %in3:  memref<64xf32>, %out3: memref<64xf32>)
      {layout = @layout} {
    csl_host.memcpy_h2d %in0 to @layout::@buf_pe0_in
        {px = 0 : i64, py = 0 : i64, width = 1 : i64, height = 1 : i64}
        : memref<64xf32>
    csl_host.memcpy_h2d %in1 to @layout::@buf_pe1_in
        {px = 1 : i64, py = 0 : i64, width = 1 : i64, height = 1 : i64}
        : memref<64xf32>
    csl_host.memcpy_h2d %in2 to @layout::@buf_pe2_in
        {px = 2 : i64, py = 0 : i64, width = 1 : i64, height = 1 : i64}
        : memref<64xf32>
    csl_host.memcpy_h2d %in3 to @layout::@buf_pe3_in
        {px = 3 : i64, py = 0 : i64, width = 1 : i64, height = 1 : i64}
        : memref<64xf32>
    csl_host.launch @layout::@compute
    csl_host.memcpy_d2h @layout::@buf_pe0_out to %out0
        {px = 0 : i64, py = 0 : i64, width = 1 : i64, height = 1 : i64}
        : memref<64xf32>
    csl_host.memcpy_d2h @layout::@buf_pe1_out to %out1
        {px = 1 : i64, py = 0 : i64, width = 1 : i64, height = 1 : i64}
        : memref<64xf32>
    csl_host.memcpy_d2h @layout::@buf_pe2_out to %out2
        {px = 2 : i64, py = 0 : i64, width = 1 : i64, height = 1 : i64}
        : memref<64xf32>
    csl_host.memcpy_d2h @layout::@buf_pe3_out to %out3
        {px = 3 : i64, py = 0 : i64, width = 1 : i64, height = 1 : i64}
        : memref<64xf32>
  }
}
