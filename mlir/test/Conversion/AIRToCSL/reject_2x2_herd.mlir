// RUN: not air-opt %s -air-to-csl 2>&1 | FileCheck %s
// CHECK: only 1x1 herds

module {
  func.func @bad(%a: memref<256xf32>) {
    %one = arith.constant 1 : index
    air.launch (%tx) in (%sx=%one) args(%la=%a) : memref<256xf32> {
      air.segment @s args(%sa=%la) : memref<256xf32> {
        %two = arith.constant 2 : index
        air.herd @h tile(%htx,%hty) in (%hsx=%two,%hsy=%two) args(%ha=%sa) : memref<256xf32> {
          air.herd_terminator
        }
        air.segment_terminator
      }
      air.launch_terminator
    }
    func.return
  }
}

