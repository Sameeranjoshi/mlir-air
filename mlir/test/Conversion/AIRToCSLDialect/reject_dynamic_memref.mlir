// RUN: not air-opt %s -air-to-csl-dialect 2>&1 | FileCheck %s
// CHECK: kernel memrefs must be statically shaped

module {
  func.func @bad(%a: memref<?xf32>) {
    %one = arith.constant 1 : index
    air.launch (%tx) in (%sx=%one) args(%la=%a) : memref<?xf32> {
      air.segment @s args(%sa=%la) : memref<?xf32> {
        %one2 = arith.constant 1 : index
        air.herd @h tile(%htx,%hty) in (%hsx=%one2,%hsy=%one2) args(%ha=%sa) : memref<?xf32> {
          air.herd_terminator
        }
        air.segment_terminator
      }
      air.launch_terminator
    }
    func.return
  }
}
