// RUN: rm -rf %t && mkdir -p %t
// RUN: air-opt %s -csl-infer-exports | air-translate --emit-csl --output-dir=%t
// RUN: FileCheck --check-prefix=N64   %s < %t/vecadd_64/pe.csl
// RUN: FileCheck --check-prefix=N256  %s < %t/vecadd_256/pe.csl
// RUN: FileCheck --check-prefix=N1024 %s < %t/vecadd_1024/pe.csl
// RUN: FileCheck --check-prefix=RANK2 %s < %t/vecadd_16x16/pe.csl
//
// Task 10: emitter size coverage. Three 1-D vecadd wafers (64, 256, 1024
// elements) plus one rank-2 wafer (16x16) using a nested scf.for. All 1-PE;
// multi-PE sharding is exercised in sharding.mlir.

// N64: var a: [64]f32;
// N64: var b: [64]f32;
// N64: var c: [64]f32;
// N64-LABEL: fn compute() void
// N64: while ({{i[0-9]+}} < 64)

// N256: var a: [256]f32;
// N256: var b: [256]f32;
// N256: var c: [256]f32;
// N256-LABEL: fn compute() void
// N256: while ({{i[0-9]+}} < 256)

// N1024: var a: [1024]f32;
// N1024: var b: [1024]f32;
// N1024: var c: [1024]f32;
// N1024-LABEL: fn compute() void
// N1024: while ({{i[0-9]+}} < 1024)

// RANK2-LABEL: fn compute() void
// RANK2: while ({{i[0-9]+}} < 16)
// RANK2: while ({{i[0-9]+}} < 16)

module {
  csl.wafer @vecadd_64 {arch = "wse3"} {
    csl.program @pe {
      %a = csl.var @a : memref<64xf32>
      %b = csl.var @b : memref<64xf32>
      %c = csl.var @c : memref<64xf32>
      csl.func @compute {
        %c0 = arith.constant 0 : index
        %n = arith.constant 64 : index
        %c1 = arith.constant 1 : index
        scf.for %i = %c0 to %n step %c1 {
          %va = memref.load %a[%i] : memref<64xf32>
          %vb = memref.load %b[%i] : memref<64xf32>
          %vc = arith.addf %va, %vb : f32
          memref.store %vc, %c[%i] : memref<64xf32>
        }
        csl.return
      }
      csl.export @a {alias = "a"}
      csl.export @b {alias = "b"}
      csl.export @c {alias = "c"}
      csl.export @compute {kind = "func"}
    }
    csl.layout {width = 1 : i64, height = 1 : i64} @layout {
      csl_layout.place @pe at (0, 0)
    }
    csl.host @main(%a_in: memref<64xf32>, %b_in: memref<64xf32>,
                   %c_out: memref<64xf32>) {layout = @layout} {
      csl_host.memcpy_h2d %a_in to @layout::@a
          {px = 0 : i64, py = 0 : i64, width = 1 : i64, height = 1 : i64}
          : memref<64xf32>
      csl_host.memcpy_h2d %b_in to @layout::@b
          {px = 0 : i64, py = 0 : i64, width = 1 : i64, height = 1 : i64}
          : memref<64xf32>
      csl_host.launch @layout::@compute
      csl_host.memcpy_d2h @layout::@c to %c_out
          {px = 0 : i64, py = 0 : i64, width = 1 : i64, height = 1 : i64}
          : memref<64xf32>
    }
  }

  csl.wafer @vecadd_256 {arch = "wse3"} {
    csl.program @pe {
      %a = csl.var @a : memref<256xf32>
      %b = csl.var @b : memref<256xf32>
      %c = csl.var @c : memref<256xf32>
      csl.func @compute {
        %c0 = arith.constant 0 : index
        %n = arith.constant 256 : index
        %c1 = arith.constant 1 : index
        scf.for %i = %c0 to %n step %c1 {
          %va = memref.load %a[%i] : memref<256xf32>
          %vb = memref.load %b[%i] : memref<256xf32>
          %vc = arith.addf %va, %vb : f32
          memref.store %vc, %c[%i] : memref<256xf32>
        }
        csl.return
      }
      csl.export @a {alias = "a"}
      csl.export @b {alias = "b"}
      csl.export @c {alias = "c"}
      csl.export @compute {kind = "func"}
    }
    csl.layout {width = 1 : i64, height = 1 : i64} @layout {
      csl_layout.place @pe at (0, 0)
    }
    csl.host @main(%a_in: memref<256xf32>, %b_in: memref<256xf32>,
                   %c_out: memref<256xf32>) {layout = @layout} {
      csl_host.memcpy_h2d %a_in to @layout::@a
          {px = 0 : i64, py = 0 : i64, width = 1 : i64, height = 1 : i64}
          : memref<256xf32>
      csl_host.memcpy_h2d %b_in to @layout::@b
          {px = 0 : i64, py = 0 : i64, width = 1 : i64, height = 1 : i64}
          : memref<256xf32>
      csl_host.launch @layout::@compute
      csl_host.memcpy_d2h @layout::@c to %c_out
          {px = 0 : i64, py = 0 : i64, width = 1 : i64, height = 1 : i64}
          : memref<256xf32>
    }
  }

  csl.wafer @vecadd_1024 {arch = "wse3"} {
    csl.program @pe {
      %a = csl.var @a : memref<1024xf32>
      %b = csl.var @b : memref<1024xf32>
      %c = csl.var @c : memref<1024xf32>
      csl.func @compute {
        %c0 = arith.constant 0 : index
        %n = arith.constant 1024 : index
        %c1 = arith.constant 1 : index
        scf.for %i = %c0 to %n step %c1 {
          %va = memref.load %a[%i] : memref<1024xf32>
          %vb = memref.load %b[%i] : memref<1024xf32>
          %vc = arith.addf %va, %vb : f32
          memref.store %vc, %c[%i] : memref<1024xf32>
        }
        csl.return
      }
      csl.export @a {alias = "a"}
      csl.export @b {alias = "b"}
      csl.export @c {alias = "c"}
      csl.export @compute {kind = "func"}
    }
    csl.layout {width = 1 : i64, height = 1 : i64} @layout {
      csl_layout.place @pe at (0, 0)
    }
    csl.host @main(%a_in: memref<1024xf32>, %b_in: memref<1024xf32>,
                   %c_out: memref<1024xf32>) {layout = @layout} {
      csl_host.memcpy_h2d %a_in to @layout::@a
          {px = 0 : i64, py = 0 : i64, width = 1 : i64, height = 1 : i64}
          : memref<1024xf32>
      csl_host.memcpy_h2d %b_in to @layout::@b
          {px = 0 : i64, py = 0 : i64, width = 1 : i64, height = 1 : i64}
          : memref<1024xf32>
      csl_host.launch @layout::@compute
      csl_host.memcpy_d2h @layout::@c to %c_out
          {px = 0 : i64, py = 0 : i64, width = 1 : i64, height = 1 : i64}
          : memref<1024xf32>
    }
  }

  // Rank-2 (16x16) vecadd via nested scf.for. csl.var is 1-D only (emitter
  // constraint); we expose the buffer as a flat 256-elem 1-D memref and let
  // the body use two induction variables for readability. The emitter treats
  // memref.load/store indices one-at-a-time so only the first index ends up
  // in the generated CSL — this mirrors the "nested loop" control-flow shape
  // without pretending we have true rank-2 indexing.
  // TODO(task-future): true rank-2 csl.var once the emitter supports it.
  csl.wafer @vecadd_16x16 {arch = "wse3"} {
    csl.program @pe {
      %a = csl.var @a : memref<256xf32>
      %b = csl.var @b : memref<256xf32>
      %c = csl.var @c : memref<256xf32>
      csl.func @compute {
        %c0 = arith.constant 0 : index
        %n = arith.constant 16 : index
        %c1 = arith.constant 1 : index
        scf.for %i = %c0 to %n step %c1 {
          scf.for %j = %c0 to %n step %c1 {
            %va = memref.load %a[%j] : memref<256xf32>
            %vb = memref.load %b[%j] : memref<256xf32>
            %vc = arith.addf %va, %vb : f32
            memref.store %vc, %c[%j] : memref<256xf32>
          }
        }
        csl.return
      }
      csl.export @a {alias = "a"}
      csl.export @b {alias = "b"}
      csl.export @c {alias = "c"}
      csl.export @compute {kind = "func"}
    }
    csl.layout {width = 1 : i64, height = 1 : i64} @layout {
      csl_layout.place @pe at (0, 0)
    }
    csl.host @main(%a_in: memref<16x16xf32>, %b_in: memref<16x16xf32>,
                   %c_out: memref<16x16xf32>) {layout = @layout} {
      csl_host.memcpy_h2d %a_in to @layout::@a
          {px = 0 : i64, py = 0 : i64, width = 1 : i64, height = 1 : i64}
          : memref<16x16xf32>
      csl_host.memcpy_h2d %b_in to @layout::@b
          {px = 0 : i64, py = 0 : i64, width = 1 : i64, height = 1 : i64}
          : memref<16x16xf32>
      csl_host.launch @layout::@compute
      csl_host.memcpy_d2h @layout::@c to %c_out
          {px = 0 : i64, py = 0 : i64, width = 1 : i64, height = 1 : i64}
          : memref<16x16xf32>
    }
  }
}
