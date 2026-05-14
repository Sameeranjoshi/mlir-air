// RUN: air-opt --csl-materialize-dataflow-colors %s | FileCheck %s
//
// User-declared `csl.color @foo_color` exists; stream `@foo` would naturally
// claim that name. The pass must uniquify the synthesized name to avoid
// SymbolTable redefinition.

csl.wafer @w {arch = "wse3"} {
  csl.program @p { csl.func @c { csl.return } }
  csl.layout {width = 2 : i64, height = 1 : i64} @layout {
    // CHECK-DAG: csl.color @foo_color{{$}}
    // CHECK-DAG: csl.color @foo_color_0{{$}}
    csl.color @foo_color
    csl_layout.dataflow @foo from(0, 0) to(1, 0)
    csl_layout.place @p at (0, 0)
    csl_layout.place @p at (1, 0)
  }
}

// CHECK: csl_layout.dataflow @foo from(0, 0) to(1, 0) {color = @foo_color_0}
