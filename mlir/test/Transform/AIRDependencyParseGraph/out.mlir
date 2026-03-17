module {
  func.func @test(%arg0: memref<256xi32>) {
    %c1 = arith.constant 1 : index
    %0 = air.herd async  tile (%arg1, %arg2) in (%arg3=%c1, %arg4=%c1) args(%arg5=%arg0) : memref<256xi32> {
      %async_token, %results = air.execute -> (memref<256xi32, 2>) {
        %alloc = memref.alloc() : memref<256xi32, 2>
        air.execute_terminator %alloc : memref<256xi32, 2>
      }
      %1 = air.dma_memcpy_nd async [%async_token] (%results[] [] [], %arg5[] [] []) : (memref<256xi32, 2>, memref<256xi32>)
      %2 = air.dma_memcpy_nd async [%1] (%arg5[] [] [], %results[] [] []) : (memref<256xi32>, memref<256xi32, 2>)
      %async_token_0 = air.execute [%2] {
        memref.dealloc %results : memref<256xi32, 2>
      }
    }
    return
  }
}

