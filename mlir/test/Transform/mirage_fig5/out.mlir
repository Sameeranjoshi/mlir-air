#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map3 = affine_map<(d0, d1, d2) -> (d1, d2)>
#map4 = affine_map<(d0, d1) -> (d0)>
module {
  func.func @attention_block(%arg0: memref<64x64xf32>, %arg1: memref<64x64xf32>, %arg2: memref<64x64xf32>, %arg3: memref<64x64xf32>, %arg4: memref<64x64xf32>, %arg5: memref<64x64xf32>, %arg6: memref<64x64xf32>, %arg7: memref<64x64xf32>, %arg8: memref<64x64xf32>, %arg9: memref<64x64xf32>) {
    %0 = air.segment async  args(%arg10=%arg0, %arg11=%arg1, %arg12=%arg2, %arg13=%arg3, %arg14=%arg4, %arg15=%arg5, %arg16=%arg6, %arg17=%arg7, %arg18=%arg8, %arg19=%arg9) : memref<64x64xf32>, memref<64x64xf32>, memref<64x64xf32>, memref<64x64xf32>, memref<64x64xf32>, memref<64x64xf32>, memref<64x64xf32>, memref<64x64xf32>, memref<64x64xf32>, memref<64x64xf32> attributes {id = 7 : i32} {
      %c2 = arith.constant 2 : index
      %c1 = arith.constant 1 : index
      %c64 = arith.constant 64 : index
      %cst = arith.constant 0.000000e+00 : f32
      %cst_0 = arith.constant 8.000000e+00 : f32
      %1 = air.herd @q_proj async  tile (%arg20, %arg21) in (%arg22=%c2, %arg23=%c1) args(%arg24=%arg14, %arg25=%arg10) : memref<64x64xf32>, memref<64x64xf32> attributes {id = 1 : i32} {
        %async_token = air.execute {
          linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%arg25 : memref<64x64xf32>) outs(%arg24 : memref<64x64xf32>) {
          ^bb0(%in: f32, %out: f32):
            linalg.yield %in : f32
          }
        } {id = 1 : i32}
      }
      %2 = air.herd @k_proj async  tile (%arg20, %arg21) in (%arg22=%c2, %arg23=%c1) args(%arg24=%arg15, %arg25=%arg11) : memref<64x64xf32>, memref<64x64xf32> attributes {id = 2 : i32} {
        %async_token = air.execute {
          linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%arg25 : memref<64x64xf32>) outs(%arg24 : memref<64x64xf32>) {
          ^bb0(%in: f32, %out: f32):
            linalg.yield %in : f32
          }
        } {id = 2 : i32}
      }
      %3 = air.herd @v_proj async  tile (%arg20, %arg21) in (%arg22=%c2, %arg23=%c1) args(%arg24=%arg16, %arg25=%arg12) : memref<64x64xf32>, memref<64x64xf32> attributes {id = 3 : i32} {
        %async_token = air.execute {
          linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%arg25 : memref<64x64xf32>) outs(%arg24 : memref<64x64xf32>) {
          ^bb0(%in: f32, %out: f32):
            linalg.yield %in : f32
          }
        } {id = 3 : i32}
      }
      %4 = air.wait_all async [%1, %2]  {id = 1 : i32}
      %5 = air.herd @attention async [%1, %2, %4]  tile (%arg20, %arg21) in (%arg22=%c2, %arg23=%c1) args(%arg24=%arg17, %arg25=%arg14, %arg26=%arg15, %arg27=%arg18) : memref<64x64xf32>, memref<64x64xf32>, memref<64x64xf32>, memref<64x64xf32> attributes {id = 4 : i32} {
        %cst_1 = arith.constant 0.000000e+00 : f32
        %cst_2 = arith.constant 8.000000e+00 : f32
        %async_token = air.execute {
          linalg.fill ins(%cst_1 : f32) outs(%arg24 : memref<64x64xf32>)
        } {id = 4 : i32}
        %async_token_3 = air.execute [%async_token] {
          linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg25, %arg26 : memref<64x64xf32>, memref<64x64xf32>) outs(%arg24 : memref<64x64xf32>) {
          ^bb0(%in: f32, %in_11: f32, %out: f32):
            %9 = arith.mulf %in, %in_11 : f32
            %10 = arith.addf %out, %9 : f32
            linalg.yield %10 : f32
          }
        } {id = 5 : i32}
        %async_token_4 = air.execute [%async_token_3] {
          linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%arg24 : memref<64x64xf32>) outs(%arg24 : memref<64x64xf32>) {
          ^bb0(%in: f32, %out: f32):
            %9 = arith.divf %in, %cst_2 : f32
            linalg.yield %9 : f32
          }
        } {id = 6 : i32}
        %async_token_5 = air.execute [%async_token_4] {
          linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%arg24 : memref<64x64xf32>) outs(%arg27 : memref<64x64xf32>) {
          ^bb0(%in: f32, %out: f32):
            %9 = math.exp %in : f32
            linalg.yield %9 : f32
          }
        } {id = 7 : i32}
        %async_token_6, %results = air.execute -> (memref<64xf32, 2 : i32>) {
          %alloc = memref.alloc() : memref<64xf32, 2 : i32>
          air.execute_terminator %alloc : memref<64xf32, 2 : i32>
        } {id = 8 : i32}
        %async_token_7 = air.execute [%async_token_6] {
          linalg.fill ins(%cst_1 : f32) outs(%results : memref<64xf32, 2 : i32>)
        } {id = 9 : i32}
        %async_token_8 = air.execute [%async_token_7, %async_token_5] {
          linalg.generic {indexing_maps = [#map, #map4], iterator_types = ["parallel", "reduction"]} ins(%arg27 : memref<64x64xf32>) outs(%results : memref<64xf32, 2 : i32>) {
          ^bb0(%in: f32, %out: f32):
            %9 = arith.addf %in, %out : f32
            linalg.yield %9 : f32
          }
        } {id = 10 : i32}
        %async_token_9 = air.execute [%async_token_8] {
          linalg.generic {indexing_maps = [#map, #map4, #map], iterator_types = ["parallel", "parallel"]} ins(%arg27, %results : memref<64x64xf32>, memref<64xf32, 2 : i32>) outs(%arg27 : memref<64x64xf32>) {
          ^bb0(%in: f32, %in_11: f32, %out: f32):
            %9 = arith.divf %in, %in_11 : f32
            linalg.yield %9 : f32
          }
        } {id = 11 : i32}
        %async_token_10 = air.execute [%async_token_9] {
          memref.dealloc %results : memref<64xf32, 2 : i32>
        } {id = 12 : i32}
      }
      %6 = air.wait_all async [%3, %5]  {id = 2 : i32}
      %7 = air.herd @out_proj async [%3, %5, %6]  tile (%arg20, %arg21) in (%arg22=%c2, %arg23=%c2) args(%arg24=%arg19, %arg25=%arg18, %arg26=%arg16) : memref<64x64xf32>, memref<64x64xf32>, memref<64x64xf32> attributes {id = 5 : i32} {
        %async_token = air.execute {
          linalg.matmul ins(%arg25, %arg26 : memref<64x64xf32>, memref<64x64xf32>) outs(%arg24 : memref<64x64xf32>)
        } {id = 13 : i32}
      }
      %8 = air.herd @rmsnorm async [%7]  tile (%arg20, %arg21) in (%arg22=%c2, %arg23=%c1) args(%arg24=%arg13, %arg25=%arg19) : memref<64x64xf32>, memref<64x64xf32> attributes {id = 6 : i32} {
        %async_token = air.execute {
          linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%arg25 : memref<64x64xf32>) outs(%arg24 : memref<64x64xf32>) {
          ^bb0(%in: f32, %out: f32):
            linalg.yield %in : f32
          }
        } {id = 14 : i32}
      }
    }
    return
  }
}

