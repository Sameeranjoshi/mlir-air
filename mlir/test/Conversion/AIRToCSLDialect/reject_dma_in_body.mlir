// RUN: not air-opt %s -air-to-csl-dialect 2>&1 | FileCheck %s
// CHECK: dma_memcpy_nd not yet supported

module {
  func.func @bad(%a: memref<256xf32>, %b: memref<256xf32>, %c: memref<256xf32>) {
    %one = arith.constant 1 : index
    air.launch (%tx) in (%sx=%one) args(%la=%a, %lb=%b, %lc=%c)
        : memref<256xf32>, memref<256xf32>, memref<256xf32> {
      air.segment @s args(%sa=%la, %sb=%lb, %sc=%lc)
          : memref<256xf32>, memref<256xf32>, memref<256xf32> {
        %one2 = arith.constant 1 : index
        air.herd @h tile(%htx,%hty) in (%hsx=%one2,%hsy=%one2)
            args(%ha=%sa, %hb=%sb, %hc=%sc) : memref<256xf32>, memref<256xf32>, memref<256xf32> {
          air.dma_memcpy_nd (%hc[][][], %ha[][][]) : (memref<256xf32>, memref<256xf32>)
          air.herd_terminator
        }
        air.segment_terminator
      }
      air.launch_terminator
    }
    func.return
  }
}
