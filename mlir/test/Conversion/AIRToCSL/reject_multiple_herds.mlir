// RUN: not air-opt %s -air-to-csl 2>&1 | FileCheck %s
// CHECK: multiple herds not yet supported

module {
  func.func @bad(%a: memref<256xf32>) {
    %one = arith.constant 1 : index
    air.launch (%tx) in (%sx=%one) args(%la=%a) : memref<256xf32> {
      air.segment @s args(%sa=%la) : memref<256xf32> {
        %one2 = arith.constant 1 : index
        air.herd @h1 tile(%htx,%hty) in (%hsx=%one2,%hsy=%one2) args(%ha=%sa) : memref<256xf32> {
          air.herd_terminator
        }
        air.herd @h2 tile(%htx2,%hty2) in (%hsx2=%one2,%hsy2=%one2) args(%ha2=%sa) : memref<256xf32> {
          air.herd_terminator
        }
        air.segment_terminator
      }
      air.launch_terminator
    }
    func.return
  }
}
