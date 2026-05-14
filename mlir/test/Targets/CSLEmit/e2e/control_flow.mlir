// RUN: rm -rf %t && mkdir -p %t
// RUN: air-opt %s -csl-infer-exports | air-translate --emit-csl --output-dir=%t
// RUN: FileCheck --check-prefix=SIMPLE  %s < %t/simple_for/pe.csl
// RUN: FileCheck --check-prefix=NESTED  %s < %t/nested_for/pe.csl
// RUN: FileCheck --check-prefix=CHAIN   %s < %t/chain_ops/pe.csl
// RUN: FileCheck --check-prefix=TEMPS   %s < %t/temps/pe.csl
//
// Task 10: exercise the scf.for/arith control-flow shapes the emitter
// supports. All wafers are 1-PE.

// SIMPLE-LABEL: fn compute() void
// SIMPLE: var {{i[0-9]+}}: u16 = 0;
// SIMPLE: while ({{i[0-9]+}} < 64)

// NESTED-LABEL: fn compute() void
// NESTED: while ({{i[0-9]+}} < 8)
// NESTED: while ({{i[0-9]+}} < 8)

// CHAIN-LABEL: fn compute() void
// CHAIN: while ({{i[0-9]+}} < 64)
// CHAIN: var {{t[0-9]+}}: f32 = {{t[0-9]+}} + {{t[0-9]+}};
// CHAIN: var {{t[0-9]+}}: f32 = {{t[0-9]+}} * {{t[0-9]+}};
// CHAIN: var {{t[0-9]+}}: f32 = {{t[0-9]+}} - {{t[0-9]+}};

// TEMPS-LABEL: fn compute() void
// TEMPS: while ({{i[0-9]+}} < 64)
// TEMPS: var {{t[0-9]+}} = a[{{i[0-9]+}}];
// TEMPS: var {{t[0-9]+}} = b[{{i[0-9]+}}];
// TEMPS: var {{t[0-9]+}}: f32 = {{t[0-9]+}} + {{t[0-9]+}};
// TEMPS: var {{t[0-9]+}}: f32 = {{t[0-9]+}} * {{t[0-9]+}};
// TEMPS: c[{{i[0-9]+}}] = {{t[0-9]+}};
// TEMPS: d[{{i[0-9]+}}] = {{t[0-9]+}};

module {
  // Simple scf.for with a trivial body.
  csl.wafer @simple_for {arch = "wse3"} {
    csl.program @pe {
      %a = csl.var @a : memref<64xf32>
      csl.func @compute {
        %c0 = arith.constant 0 : index
        %n = arith.constant 64 : index
        %c1 = arith.constant 1 : index
        scf.for %i = %c0 to %n step %c1 {
          %va = memref.load %a[%i] : memref<64xf32>
          memref.store %va, %a[%i] : memref<64xf32>
        }
        csl.return
      }
      csl.export @a {alias = "a"}
      csl.export @compute {kind = "func"}
    }
    csl.layout {width = 1 : i64, height = 1 : i64} @layout {
      csl_layout.place @pe at (0, 0)
    }
    csl.host @main(%a: memref<64xf32>) {layout = @layout} {
      csl_host.memcpy_h2d %a to @layout::@a
          {px = 0 : i64, py = 0 : i64, width = 1 : i64, height = 1 : i64}
          : memref<64xf32>
      csl_host.launch @layout::@compute
    }
  }

  // Nested scf.for — 2 levels.
  csl.wafer @nested_for {arch = "wse3"} {
    csl.program @pe {
      %m = csl.var @m : memref<64xf32>
      csl.func @compute {
        %c0 = arith.constant 0 : index
        %n = arith.constant 8 : index
        %c1 = arith.constant 1 : index
        scf.for %i = %c0 to %n step %c1 {
          scf.for %j = %c0 to %n step %c1 {
            %vm = memref.load %m[%j] : memref<64xf32>
            memref.store %vm, %m[%j] : memref<64xf32>
          }
        }
        csl.return
      }
      csl.export @m {alias = "m"}
      csl.export @compute {kind = "func"}
    }
    csl.layout {width = 1 : i64, height = 1 : i64} @layout {
      csl_layout.place @pe at (0, 0)
    }
    csl.host @main(%m: memref<64xf32>) {layout = @layout} {
      csl_host.memcpy_h2d %m to @layout::@m
          {px = 0 : i64, py = 0 : i64, width = 1 : i64, height = 1 : i64}
          : memref<64xf32>
      csl_host.launch @layout::@compute
    }
  }

  // 3-arith-op chain: addf -> mulf -> subf in the same body.
  csl.wafer @chain_ops {arch = "wse3"} {
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
          %s = arith.addf %va, %vb : f32
          %p = arith.mulf %s, %va : f32
          %r = arith.subf %p, %vb : f32
          memref.store %r, %c[%i] : memref<64xf32>
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
    csl.host @main(%a: memref<64xf32>, %b: memref<64xf32>, %c: memref<64xf32>)
        {layout = @layout} {
      csl_host.memcpy_h2d %a to @layout::@a
          {px = 0 : i64, py = 0 : i64, width = 1 : i64, height = 1 : i64}
          : memref<64xf32>
      csl_host.memcpy_h2d %b to @layout::@b
          {px = 0 : i64, py = 0 : i64, width = 1 : i64, height = 1 : i64}
          : memref<64xf32>
      csl_host.launch @layout::@compute
      csl_host.memcpy_d2h @layout::@c to %c
          {px = 0 : i64, py = 0 : i64, width = 1 : i64, height = 1 : i64}
          : memref<64xf32>
    }
  }

  // Multiple loads and stores intermixed in the same loop body, with temps.
  csl.wafer @temps {arch = "wse3"} {
    csl.program @pe {
      %a = csl.var @a : memref<64xf32>
      %b = csl.var @b : memref<64xf32>
      %c = csl.var @c : memref<64xf32>
      %d = csl.var @d : memref<64xf32>
      csl.func @compute {
        %c0 = arith.constant 0 : index
        %n = arith.constant 64 : index
        %c1 = arith.constant 1 : index
        scf.for %i = %c0 to %n step %c1 {
          %va = memref.load %a[%i] : memref<64xf32>
          %vb = memref.load %b[%i] : memref<64xf32>
          %sum  = arith.addf %va, %vb : f32
          %prod = arith.mulf %va, %vb : f32
          memref.store %sum,  %c[%i] : memref<64xf32>
          memref.store %prod, %d[%i] : memref<64xf32>
        }
        csl.return
      }
      csl.export @a {alias = "a"}
      csl.export @b {alias = "b"}
      csl.export @c {alias = "c"}
      csl.export @d {alias = "d"}
      csl.export @compute {kind = "func"}
    }
    csl.layout {width = 1 : i64, height = 1 : i64} @layout {
      csl_layout.place @pe at (0, 0)
    }
    csl.host @main(%a: memref<64xf32>, %b: memref<64xf32>,
                   %c: memref<64xf32>, %d: memref<64xf32>)
        {layout = @layout} {
      csl_host.memcpy_h2d %a to @layout::@a
          {px = 0 : i64, py = 0 : i64, width = 1 : i64, height = 1 : i64}
          : memref<64xf32>
      csl_host.memcpy_h2d %b to @layout::@b
          {px = 0 : i64, py = 0 : i64, width = 1 : i64, height = 1 : i64}
          : memref<64xf32>
      csl_host.launch @layout::@compute
      csl_host.memcpy_d2h @layout::@c to %c
          {px = 0 : i64, py = 0 : i64, width = 1 : i64, height = 1 : i64}
          : memref<64xf32>
      csl_host.memcpy_d2h @layout::@d to %d
          {px = 0 : i64, py = 0 : i64, width = 1 : i64, height = 1 : i64}
          : memref<64xf32>
    }
  }
}
