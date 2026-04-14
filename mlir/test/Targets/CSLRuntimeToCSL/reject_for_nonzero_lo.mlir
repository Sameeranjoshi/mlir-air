// RUN: not air-translate --emit-csl-rt --csl-output-dir=%t %s -o /dev/null 2>&1 | FileCheck %s

// CHECK: scf.for requires lo=0

module {
  func.func @bad_for() {
    csl.spatial_placement {
      %k = csl.kernel {
        %x = csl.var @x : memref<256xf32>
        csl.func @bad {
          %c1 = arith.constant 1 : index
          %c256 = arith.constant 256 : index
          %step = arith.constant 1 : index
          scf.for %i = %c1 to %c256 step %step {
          }
          csl.return
        }
      } {source_file = "bad.csl"} : !csl.kernel
      %r = csl.code_region routes() colors() {
      }{width = 1 : i64, height = 1 : i64} : !csl.code_region
      csl.place %r %k {x = 0 : i64, y = 0 : i64}
    }
    csl.export_name "bad" : () -> ()
    return
  }
}
