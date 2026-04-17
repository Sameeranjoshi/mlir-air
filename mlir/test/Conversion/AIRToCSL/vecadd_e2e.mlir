// End-to-end pipeline test: AIR vecadd -> csl.wafer v2 -> three emitters.
//
// RUN: air-opt %s -air-to-csl -csl-infer-exports | air-translate --emit-csl-program | FileCheck %s --check-prefix=PROG
// RUN: air-opt %s -air-to-csl | air-translate --emit-csl-layout | FileCheck %s --check-prefix=LAYOUT
// RUN: air-opt %s -air-to-csl -csl-infer-exports | air-translate --emit-csl-host | FileCheck %s --check-prefix=HOST

// ---- PE program checks ----
// PROG: param memcpy_params: comptime_struct;
// PROG: const sys_mod = @import_module("<memcpy/memcpy>", memcpy_params);
// PROG: var arg0: [256]f32;
// PROG: var arg1: [256]f32;
// PROG: var arg2: [256]f32;
// PROG: var arg0_ptr: [*]f32 = &arg0;
// PROG: var arg1_ptr: [*]f32 = &arg1;
// PROG: const arg2_ptr: [*]f32 = &arg2;
// PROG: fn compute() void {
// PROG:   var {{.*}}: u16 = 0;
// PROG:   while ({{.*}} < 256) : ({{.*}} += 1) {
// PROG:     var {{.*}} = arg0[{{.*}}];
// PROG:     var {{.*}} = arg1[{{.*}}];
// PROG:     var {{.*}}: f32 = {{.*}} + {{.*}};
// PROG:     arg2[{{.*}}] = {{.*}};
// PROG:   }
// PROG:   sys_mod.unblock_cmd_stream();
// PROG: }
// PROG: comptime {
// PROG:   @export_symbol(arg0_ptr, "arg0");
// PROG:   @export_symbol(arg1_ptr, "arg1");
// PROG:   @export_symbol(compute);
// PROG:   @export_symbol(arg2_ptr, "arg2");
// PROG: }

// ---- Layout checks ----
// LAYOUT: from cerebras.sdk.runtime.sdkruntimepybind import SdkLayout, SdkTarget
// LAYOUT: def get_layout(target: SdkTarget) -> SdkLayout:
// LAYOUT:     layout = SdkLayout(target)
// LAYOUT:     region = layout.create_code_region("h.csl", "h", 1, 1)
// LAYOUT:     region.place(0, 0)
// LAYOUT:     return layout

// ---- Host checks ----
// HOST: #!/usr/bin/env cs_python
// HOST: import argparse
// HOST: from cerebras.sdk.runtime.sdkruntimepybind import (
// HOST:     SdkRuntime,
// HOST:     MemcpyOrder,
// HOST:     MemcpyDataType,
// HOST: )
// HOST: N = 256
// HOST: arg0 = np.arange(N, dtype=np.float32)
// HOST: arg1 = np.arange(N, dtype=np.float32) * 2.0
// HOST: arg2 = np.zeros(N, dtype=np.float32)
// HOST: runner = SdkRuntime(args.name, cmaddr=args.cmaddr)
// HOST: runner.load()
// HOST: runner.run()
// HOST: runner.memcpy_h2d(runner.get_id("arg0"), arg0, 0, 0, 1, 1, 256,
// HOST: runner.memcpy_h2d(runner.get_id("arg1"), arg1, 0, 0, 1, 1, 256,
// HOST: runner.launch("compute", nonblock=False)
// HOST: runner.memcpy_d2h(arg2, runner.get_id("arg2"), 0, 0, 1, 1, 256,
// HOST: runner.stop()
// HOST: print("SUCCESS!")

module {
  func.func @vecadd(%a: memref<256xf32>, %b: memref<256xf32>,
                    %c: memref<256xf32>) {
    %one = arith.constant 1 : index
    air.launch (%tx) in (%sx=%one) args(%la=%a, %lb=%b, %lc=%c)
        : memref<256xf32>, memref<256xf32>, memref<256xf32> {
      air.segment @seg0 args(%sa=%la, %sb=%lb, %sc=%lc)
          : memref<256xf32>, memref<256xf32>, memref<256xf32> {
        %one2 = arith.constant 1 : index
        air.herd @h tile(%htx, %hty) in (%hsx=%one2, %hsy=%one2)
            args(%ha=%sa, %hb=%sb, %hc=%sc)
            : memref<256xf32>, memref<256xf32>, memref<256xf32> {
          %lo = arith.constant 0 : index
          %hi = arith.constant 256 : index
          %step = arith.constant 1 : index
          scf.for %i = %lo to %hi step %step {
            %va = memref.load %ha[%i] : memref<256xf32>
            %vb = memref.load %hb[%i] : memref<256xf32>
            %vc = arith.addf %va, %vb : f32
            memref.store %vc, %hc[%i] : memref<256xf32>
          }
          air.herd_terminator
        }
        air.segment_terminator
      }
      air.launch_terminator
    }
    return
  }
}
