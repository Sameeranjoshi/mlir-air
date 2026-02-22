//===- basic.mlir -  AIR to CSL basic lowering test ------------*- MLIR -*-===//

// RUN: air-opt %s -air-to-csl="output-dir=%t" && cat %t/layout.csl | FileCheck %s --check-prefix=LAYOUT
// RUN: cat %t/pe_program.csl | FileCheck %s --check-prefix=PE
// RUN: cat %t/run.py | FileCheck %s --check-prefix=RUN

// A minimal AIR program: 1x1 herd that adds two values.

// LAYOUT: @set_rectangle(1, 1)
// LAYOUT: @set_tile_code(0, 0, "pe_program.csl"
// LAYOUT: @export_name("init_and_compute", fn()void)

// PE: param memcpy_params: comptime_struct
// PE: sys_mod
// PE: fn compute()
// PE: fn init_and_compute()
// PE: sys_mod.unblock_cmd_stream()
// PE: comptime
// PE: @export_symbol(init_and_compute)

// RUN: SdkRuntime
// RUN: runner.load()
// RUN: runner.run()
// RUN: runner.launch('init_and_compute'
// RUN: runner.stop()

module {
  func.func @vecadd(%a: memref<1024xf32>, %b: memref<1024xf32>, %c: memref<1024xf32>) {
    %c1 = arith.constant 1 : index
    air.launch (%tx) in (%sx=%c1) args(%arg0=%a, %arg1=%b, %arg2=%c) : memref<1024xf32>, memref<1024xf32>, memref<1024xf32> {
      air.segment @seg0 args(%sarg0=%arg0, %sarg1=%arg1, %sarg2=%arg2) : memref<1024xf32>, memref<1024xf32>, memref<1024xf32> {
        %c1_0 = arith.constant 1 : index
        air.herd @herd0 tile(%htx, %hty) in (%hsx=%c1_0, %hsy=%c1_0) args(%harg0=%sarg0, %harg1=%sarg1, %harg2=%sarg2) : memref<1024xf32>, memref<1024xf32>, memref<1024xf32> {
          %zero = arith.constant 0 : index
          %v0 = memref.load %harg0[%zero] : memref<1024xf32>
          %v1 = memref.load %harg1[%zero] : memref<1024xf32>
          %v2 = arith.addf %v0, %v1 : f32
          memref.store %v2, %harg2[%zero] : memref<1024xf32>
          air.herd_terminator
        }
        air.segment_terminator
      }
      air.launch_terminator
    }
    return
  }
}
