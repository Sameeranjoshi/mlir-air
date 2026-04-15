// End-to-end pipeline test: AIR vecadd -> csl.wafer v2 -> three emitters.
//
// RUN: air-opt %s -air-to-csl -csl-derive-exports | air-translate --emit-csl-program | FileCheck %s --check-prefix=PROG
// RUN: air-opt %s -air-to-csl | air-translate --emit-csl-layout | FileCheck %s --check-prefix=LAYOUT
// RUN: air-opt %s -air-to-csl -csl-derive-exports | air-translate --emit-csl-host | FileCheck %s --check-prefix=HOST

// ---- PE program checks ----
// PROG: var arg0: [256]f32;
// PROG: var arg1: [256]f32;
// PROG: var arg2: [256]f32;
// PROG: fn compute() void {
// PROG:   var {{.*}}: u16 = 0;
// PROG:   while ({{.*}} < 256) : ({{.*}} += 1) {
// PROG:     var {{.*}} = arg0[{{.*}}];
// PROG:     var {{.*}} = arg1[{{.*}}];
// PROG:     var {{.*}}: f32 = {{.*}} + {{.*}};
// PROG:     arg2[{{.*}}] = {{.*}};
// PROG:   }
// PROG: }
// PROG: comptime {
// PROG:   @export_symbol(&arg0, "arg0");
// PROG:   @export_symbol(&arg1, "arg1");
// PROG:   @export_symbol(&arg2, "arg2");
// PROG:   @export_symbol(&compute, "compute");
// PROG: }

// ---- Layout checks ----
// LAYOUT: from cerebras.sdk.runtime.sdkruntimepybind import SdkLayout, SdkTarget
// LAYOUT: def get_layout(target: SdkTarget) -> SdkLayout:
// LAYOUT:     layout = SdkLayout(target)
// LAYOUT:     region = layout.create_code_region("h.csl", "h", 1, 1)
// LAYOUT:     region.place(0, 0)
// LAYOUT:     return layout

// ---- Host checks ----
// HOST: from csl_layout import get_layout
// HOST: def main(target, arg0, arg1, arg2):
// HOST:     artifacts = get_layout(target).compile("out/")
// HOST:     N = 256
// HOST:     with SdkRuntime(artifacts) as runner:
// HOST:         runner.memcpy_h2d(runner.get_id("arg0"), arg0, 0, 0, 1, 1, N)
// HOST:         runner.memcpy_h2d(runner.get_id("arg1"), arg1, 0, 0, 1, 1, N)
// HOST:         runner.launch("compute", nonblock=False)
// HOST:         runner.memcpy_d2h(arg2, runner.get_id("arg2"), 0, 0, 1, 1, N)

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
