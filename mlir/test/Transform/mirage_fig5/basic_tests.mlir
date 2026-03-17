// vector add elementwise
module {
  func.func @vec_add(
      %A: memref<4xf32>,
      %B: memref<4xf32>,
      %C: memref<4xf32>) {

    %c0 = arith.constant 0 : index
    %c4 = arith.constant 4 : index
    %c1 = arith.constant 1 : index

    scf.for %i = %c0 to %c4 step %c1 {
      %a = memref.load %A[%i] : memref<4xf32>
      %b = memref.load %B[%i] : memref<4xf32>
      %sum = arith.addf %a, %b : f32
      memref.store %sum, %C[%i] : memref<4xf32>
    }

    return
  }
}