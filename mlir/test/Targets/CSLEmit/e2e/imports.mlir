// RUN: rm -rf %t && mkdir -p %t
// RUN: air-opt %s -csl-infer-exports | air-translate --emit-csl --output-dir=%t
// RUN: FileCheck %s < %t/imports/pe.csl
//
// v5 Task 6 — emit user-level csl.import_module generically.
// The two hardcoded imports (<memcpy/memcpy>, <layout>) stay at the top.
// User-authored imports follow, in IR order, with or without params.
// A subsequent csl.builtin_call in %mod (...) references the emitted name.

// The two hardcoded imports come first.
// CHECK: @import_module("<memcpy/memcpy>"
// CHECK: @import_module("<layout>")

// Then the user imports, in IR order.
// CHECK: const {{.*}} = @import_module("<math>");
// CHECK: const {{.*}} = @import_module("<memcpy/get_params>", .{ .width = 4 });

// The builtin_call with a module operand references one of the emitted names.
// CHECK: fn compute() void
// CHECK: .sqrt(

module {
  csl.wafer @imports {arch = "wse3"} {
    csl.program @pe {
      %math = csl.import_module "<math>" : !csl.imported_module
      %gp   = csl.import_module "<memcpy/get_params>"
                  {params = {width = 4 : i32}} : !csl.imported_module
      csl.func @compute {
        %x = arith.constant 4.0 : f32
        %r = csl.builtin_call "sqrt" in %math (%x) : (f32) -> f32
        csl.return
      }
      csl.export @compute {kind = "func"}
    }
    csl.layout {width = 1 : i64, height = 1 : i64} @layout {
      csl_layout.place @pe at (0, 0)
    }
    csl.host @main() {layout = @layout} {
      csl_host.launch @layout::@compute
    }
  }
}
