// RUN: not air-opt %s -air-to-csl 2>&1 | FileCheck %s
// CHECK: inter-PE channels not yet supported

module {
  air.channel @ch [1, 1]
  func.func @bad(%a: memref<256xf32>) {
    %one = arith.constant 1 : index
    air.launch (%tx) in (%sx=%one) args(%la=%a) : memref<256xf32> {
      %zero = arith.constant 0 : index
      air.channel.put @ch[%zero, %zero] (%la[] [] []) : (memref<256xf32>)
      air.launch_terminator
    }
    func.return
  }
}
