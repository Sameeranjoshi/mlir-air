
module {
// Using Linalg dialect and performing addition
  func.func @vec_add_linalg(%A: tensor<4xf32>, %B: tensor<4xf32>) -> tensor<4xf32> {
    %init = tensor.empty() : tensor<4xf32>
    %C = linalg.generic
      {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
       iterator_types = ["parallel"]
        }
      ins(%A, %B : tensor<4xf32>, tensor<4xf32>)
      outs(%init : tensor<4xf32>) {
      ^bb0(%a: f32, %b: f32, %out: f32):
        %sum = arith.addf %a, %b : f32
        linalg.yield %sum : f32
    } -> tensor<4xf32>
    return %C : tensor<4xf32>
  }
}