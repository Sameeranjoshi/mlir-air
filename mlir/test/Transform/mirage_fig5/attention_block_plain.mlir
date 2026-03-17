//===- attention_block_plain.mlir - Attention (no AIR constructs) -*- MLIR -*-===//
//
// Algorithm:
// 1. S = Q * K^T
// 2. Scale: S = S / sqrt(d_k)
// 3. Softmax: S2 = exp(S) / sum(exp(S)) per row
// 4. attention_result = S2 * V
//
//===----------------------------------------------------------------------===//

module {
  func.func @attention_block(
      %Q: memref<64x64xf32>, %K: memref<64x64xf32>, %V: memref<64x64xf32>,
      %scores: memref<64x64xf32>, %softmax_out: memref<64x64xf32>,
      %out: memref<64x64xf32>) {
    %zero = arith.constant 0.000000e+00 : f32
    %scale = arith.constant 8.000000e+00 : f32  // sqrt(64)

    // 1. S = Q * K^T  (scores[i,j] = sum_k Q[i,k] * K[j,k])
    linalg.fill ins(%zero : f32) outs(%scores : memref<64x64xf32>)
    linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1, d2) -> (d0, d1)>,
        affine_map<(d0, d1, d2) -> (d0, d2)>,
        affine_map<(d0, d1, d2) -> (d1, d2)>
      ],
      iterator_types = ["parallel", "parallel", "reduction"]
    } ins(%Q, %K : memref<64x64xf32>, memref<64x64xf32>)
      outs(%scores : memref<64x64xf32>) {
    ^bb0(%q_val: f32, %k_val: f32, %acc: f32):
      %prod = arith.mulf %q_val, %k_val : f32
      %sum = arith.addf %acc, %prod : f32
      linalg.yield %sum : f32
    }

    // 2. Scale: S = S / sqrt(d_k)
    linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1) -> (d0, d1)>,
        affine_map<(d0, d1) -> (d0, d1)>
      ],
      iterator_types = ["parallel", "parallel"]
    } ins(%scores : memref<64x64xf32>) outs(%scores : memref<64x64xf32>) {
    ^bb0(%in_val: f32, %out_val: f32):
      %scaled = arith.divf %in_val, %scale : f32
      linalg.yield %scaled : f32
    }

    // 3. Softmax: exp then divide by row sum
    linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1) -> (d0, d1)>,
        affine_map<(d0, d1) -> (d0, d1)>
      ],
      iterator_types = ["parallel", "parallel"]
    } ins(%scores : memref<64x64xf32>) outs(%softmax_out : memref<64x64xf32>) {
    ^bb0(%in_val: f32, %out_val: f32):
      %exp_val = math.exp %in_val : f32
      linalg.yield %exp_val : f32
    }

    %row_sum = memref.alloc() : memref<64xf32>
    linalg.fill ins(%zero : f32) outs(%row_sum : memref<64xf32>)
    linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1) -> (d0, d1)>,
        affine_map<(d0, d1) -> (d0)>
      ],
      iterator_types = ["parallel", "reduction"]
    } ins(%softmax_out : memref<64x64xf32>) outs(%row_sum : memref<64xf32>) {
    ^bb0(%in_val: f32, %acc: f32):
      %sum = arith.addf %in_val, %acc : f32
      linalg.yield %sum : f32
    }

    linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1) -> (d0, d1)>,
        affine_map<(d0, d1) -> (d0)>,
        affine_map<(d0, d1) -> (d0, d1)>
      ],
      iterator_types = ["parallel", "parallel"]
    } ins(%softmax_out, %row_sum : memref<64x64xf32>, memref<64xf32>)
      outs(%softmax_out : memref<64x64xf32>) {
    ^bb0(%in_val: f32, %sum_val: f32, %out_val: f32):
      %norm = arith.divf %in_val, %sum_val : f32
      linalg.yield %norm : f32
    }

    memref.dealloc %row_sum : memref<64xf32>

    // 4. attention_result = S2 * V
    linalg.matmul ins(%softmax_out, %V : memref<64x64xf32>, memref<64x64xf32>)
                  outs(%out : memref<64x64xf32>)

    return
  }
}
