//===- attention_block.mlir - Attention block (Mirage Figure 5) -*- MLIR -*-===//
//
// Algorithm:
// Let Q, K, V be Query, Key and Value matrix respectively.
// Attention (not MHA):
// 1. Compute scores(S) = Q * K^T
// 2. Scale(S1) = S / sqrt(d_k)
// 3. Softmax(S2) = Softmax(S1)
// 4. attention_result = S2 * V
//
// Dependency structure (Mirage tGraph):
//   q_proj, k_proj, v_proj (parallel, no deps)
//   attention (scores + scale + softmax) depends on q_proj AND k_proj
//   out_proj (S2*V) depends on attention AND v_proj
//   rmsnorm depends on out_proj
//
//===----------------------------------------------------------------------===//

module {
  func.func @attention_block(
      %A: memref<64x64xf32>, %B: memref<64x64xf32>, %C: memref<64x64xf32>,
      %D: memref<64x64xf32>, %Q: memref<64x64xf32>, %K: memref<64x64xf32>,
      %V: memref<64x64xf32>, %scores: memref<64x64xf32>,
      %softmax_out: memref<64x64xf32>, %out: memref<64x64xf32>) {
    air.segment args(%a = %A, %b = %B, %c = %C, %d = %D,
                     %q_buf = %Q, %k_buf = %K, %v_buf = %V,
                     %scores_buf = %scores, %softmax_buf = %softmax_out,
                     %out_buf = %out) :
      memref<64x64xf32>, memref<64x64xf32>, memref<64x64xf32>, memref<64x64xf32>,
      memref<64x64xf32>, memref<64x64xf32>, memref<64x64xf32>,
      memref<64x64xf32>, memref<64x64xf32>, memref<64x64xf32> {
      %c2 = arith.constant 2 : index
      %c1 = arith.constant 1 : index
      %c64 = arith.constant 64 : index
      %cst_0 = arith.constant 0.000000e+00 : f32
      %cst_8 = arith.constant 8.000000e+00 : f32  // sqrt(64)

      // Q_proj (2x1): copy A -> Q (identity projection for test)
      %tok_q = air.herd @q_proj async tile(%tx, %ty) in (%sx = %c2, %sy = %c1) args(%dst = %q_buf, %src = %a) : memref<64x64xf32>, memref<64x64xf32> {
        linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%src : memref<64x64xf32>) outs(%dst : memref<64x64xf32>) {
        ^bb0(%in_val: f32, %out_val: f32):
          linalg.yield %in_val : f32
        }
        air.herd_terminator
      }

      // K_proj (2x1): copy B -> K
      %tok_k = air.herd @k_proj async tile(%tx, %ty) in (%sx = %c2, %sy = %c1) args(%dst = %k_buf, %src = %b) : memref<64x64xf32>, memref<64x64xf32> {
        linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%src : memref<64x64xf32>) outs(%dst : memref<64x64xf32>) {
        ^bb0(%in_val: f32, %out_val: f32):
          linalg.yield %in_val : f32
        }
        air.herd_terminator
      }

      // V_proj (2x1): copy C -> V
      %tok_v = air.herd @v_proj async tile(%tx, %ty) in (%sx = %c2, %sy = %c1) args(%dst = %v_buf, %src = %c) : memref<64x64xf32>, memref<64x64xf32> {
        linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%src : memref<64x64xf32>) outs(%dst : memref<64x64xf32>) {
        ^bb0(%in_val: f32, %out_val: f32):
          linalg.yield %in_val : f32
        }
        air.herd_terminator
      }

      // Attention (2x1): S = Q*K^T, scale, softmax. Depends on q_proj AND k_proj.
      %tok_qk = air.wait_all async [%tok_q, %tok_k]
      %tok_attn = air.herd @attention async [%tok_qk] tile(%tx, %ty) in (%sx = %c2, %sy = %c1) args(%attn_scores = %scores_buf, %attn_q = %q_buf, %attn_k = %k_buf, %attn_softmax = %softmax_buf) : memref<64x64xf32>, memref<64x64xf32>, memref<64x64xf32>, memref<64x64xf32> {
        %zero = arith.constant 0.000000e+00 : f32
        %scale = arith.constant 8.000000e+00 : f32  // sqrt(64)
        // 1. S = Q * K^T  (scores[i,j] = sum_k Q[i,k] * K[j,k])
        linalg.fill ins(%zero : f32) outs(%attn_scores : memref<64x64xf32>)
        linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%attn_q, %attn_k : memref<64x64xf32>, memref<64x64xf32>) outs(%attn_scores : memref<64x64xf32>) {
        ^bb0(%q_val: f32, %k_val: f32, %acc: f32):
          %prod = arith.mulf %q_val, %k_val : f32
          %sum = arith.addf %acc, %prod : f32
          linalg.yield %sum : f32
        }
        // 2. Scale: S = S / sqrt(d_k)
        linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%attn_scores : memref<64x64xf32>) outs(%attn_scores : memref<64x64xf32>) {
        ^bb0(%in_val: f32, %out_val: f32):
          %scaled = arith.divf %in_val, %scale : f32
          linalg.yield %scaled : f32
        }
        // 3. Softmax: softmax(x) = exp(x) / sum(exp(x)) per row
        //    Simplified: exp then divide by row sum
        linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%attn_scores : memref<64x64xf32>) outs(%attn_softmax : memref<64x64xf32>) {
        ^bb0(%in_val: f32, %out_val: f32):
          %exp_val = math.exp %in_val : f32
          linalg.yield %exp_val : f32
        }
        %row_sum = memref.alloc() : memref<64xf32, 2 : i32>
        linalg.fill ins(%zero : f32) outs(%row_sum : memref<64xf32, 2 : i32>)
        linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>], iterator_types = ["parallel", "reduction"]} ins(%attn_softmax : memref<64x64xf32>) outs(%row_sum : memref<64xf32, 2 : i32>) {
        ^bb0(%in_val: f32, %acc: f32):
          %sum = arith.addf %in_val, %acc : f32
          linalg.yield %sum : f32
        }
        linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%attn_softmax, %row_sum : memref<64x64xf32>, memref<64xf32, 2 : i32>) outs(%attn_softmax : memref<64x64xf32>) {
        ^bb0(%in_val: f32, %sum_val: f32, %out_val: f32):
          %norm = arith.divf %in_val, %sum_val : f32
          linalg.yield %norm : f32
        }
        memref.dealloc %row_sum : memref<64xf32, 2 : i32>
        air.herd_terminator
      }

      // Out_proj (2x2): attention_result = S2 * V. Depends on attention AND v_proj.
      %tok_av = air.wait_all async [%tok_attn, %tok_v]
      %tok_out = air.herd @out_proj async [%tok_av] tile(%tx, %ty) in (%sx = %c2, %sy = %c2) args(%dst = %out_buf, %attn = %softmax_buf, %v = %v_buf) : memref<64x64xf32>, memref<64x64xf32>, memref<64x64xf32> {
        linalg.matmul ins(%attn, %v : memref<64x64xf32>, memref<64x64xf32>) outs(%dst : memref<64x64xf32>)
        air.herd_terminator
      }

      // RMSNorm (2x1): depends on Out_proj
      %tok_rms = air.herd @rmsnorm async [%tok_out] tile(%tx, %ty) in (%sx = %c2, %sy = %c1) args(%dst = %d, %src = %out_buf) : memref<64x64xf32>, memref<64x64xf32> {
        linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%src : memref<64x64xf32>) outs(%dst : memref<64x64xf32>) {
        ^bb0(%in_val: f32, %out_val: f32):
          linalg.yield %in_val : f32
        }
        air.herd_terminator
      }

      air.segment_terminator
    }
    return
  }
}
