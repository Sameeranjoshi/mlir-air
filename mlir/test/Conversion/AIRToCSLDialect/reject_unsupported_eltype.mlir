// RUN: not air-opt %s -air-to-csl-dialect 2>&1 | FileCheck %s
// CHECK: unsupported element type

module {
  func.func @bad(%a: memref<256xf64>) {
    %one = arith.constant 1 : index
    air.launch (%tx) in (%sx=%one) args(%la=%a) : memref<256xf64> {
      air.segment @s args(%sa=%la) : memref<256xf64> {
        %one2 = arith.constant 1 : index
        air.herd @h tile(%htx,%hty) in (%hsx=%one2,%hsy=%one2) args(%ha=%sa) : memref<256xf64> {
          air.herd_terminator
        }
        air.segment_terminator
      }
      air.launch_terminator
    }
    func.return
  }
}
