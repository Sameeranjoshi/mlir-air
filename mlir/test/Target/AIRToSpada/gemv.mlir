// GEMV (examples/gemv/gemv_air.mlir) after
// `air-opt -air-place-herds-by-token="fabric-x=4 fabric-y=4"`, committed here
// as the air-to-spada lit test input. See docs/SPADA_EMITTER_SPEC.md §4.
//
// RUN: air-translate --air-to-spada %s | FileCheck %s

// Kernel signature: one readonly stream per scattered L3 input, one
// writeonly stream for the gathered result.
// CHECK: kernel @gemv<>(stream<f32, 64>[4, 4] readonly arg0_in,
// CHECK: stream<f32, 8>[4, 1] readonly arg1_in,
// CHECK: stream<f32, 8>[1, 4] writeonly arg2_out)

// The @stage herd's column multicast of x.
// CHECK: stream<f32> bx = relative_stream(0, [1:4]) { hops = auto, channel = 0 }

// The @compute herd's inner-product loop. Spec v2 §1: A is ranked (never a
// channel-get destination) and indexed per-dimension, not flattened.
// CHECK: for i16 k in [0:8] {
// CHECK: for i16 l in [0:8] {
// CHECK: {{.*}} = ({{.*}} + ({{.*}}[k, l] * {{.*}}[l]))

// The @reduce herd's westward chain reduction: two streams, colour by parity.
// CHECK: stream<f32> red_even = relative_stream(-1, 0) { hops = [(-1, 0)], channel = 1 }
// CHECK: stream<f32> red_odd = relative_stream(-1, 0) { hops = [(-1, 0)], channel = 2 }

// Strided role rectangles for the interior of the reduction chain.
// CHECK: compute i16 x, i16 y in [1, 0:4]
// CHECK: compute i16 x, i16 y in [2, 0:4]

#map = affine_map<(d0, d1) -> (d1, d0)>
#map1 = affine_map<(d0) -> (d0, 0)>
#map2 = affine_map<(d0) -> (0, d0)>
#map3 = affine_map<(d0, d1, d2) -> (d0, d1)>
#map4 = affine_map<(d0) -> (d0 * 8)>
#map5 = affine_map<(d0) -> (d0 - 1)>
#set = affine_set<(d0, d1) : (d1 == 0)>
#set1 = affine_set<(d0, d1) : (d0 - 3 == 0)>
#set2 = affine_set<(d0, d1) : (d0 == 0)>
module {
  air.channel @bx [4, 1] {broadcast_shape = [4, 4]}
  air.channel @red [4, 4]
  func.func @gemv(%arg0: memref<32x32xf32>, %arg1: memref<32xf32>, %arg2: memref<32xf32>) {
    %c1 = arith.constant 1 : index
    air.launch (%arg3) in (%arg4=%c1) args(%arg5=%arg0, %arg6=%arg1, %arg7=%arg2) : memref<32x32xf32>, memref<32xf32>, memref<32xf32> {
      air.segment @gemv_seg  args(%arg8=%arg5, %arg9=%arg6, %arg10=%arg7) : memref<32x32xf32>, memref<32xf32>, memref<32xf32> {
        %c0 = arith.constant 0 : index
        %c1_0 = arith.constant 1 : index
        %c4 = arith.constant 4 : index
        %c8 = arith.constant 8 : index
        %c32 = arith.constant 32 : index
        %c64 = arith.constant 64 : index
        %alloc = memref.alloc() {air.partition = #air.partition<block = [8, 8], owner = #map>} : memref<32x32xf32, 1>
        %alloc_1 = memref.alloc() {air.partition = #air.partition<block = [8], owner = #map1>} : memref<32xf32, 1>
        %alloc_2 = memref.alloc() {air.partition = #air.partition<block = [8], owner = #map2>} : memref<32xf32, 1>
        %alloc_3 = memref.alloc() {air.partition = #air.partition<block = [1, 1, 8], owner = #map3>} : memref<4x4x8xf32, 1>
        %alloc_4 = memref.alloc() {air.partition = #air.partition<block = [1, 1, 8], owner = #map3>} : memref<4x4x8xf32, 1>
        air.dma_memcpy_nd (%alloc[] [] [], %arg8[] [] []) : (memref<32x32xf32, 1>, memref<32x32xf32>)
        air.dma_memcpy_nd (%alloc_1[] [] [], %arg9[] [] []) : (memref<32xf32, 1>, memref<32xf32>)
        %0 = air.token.alloc : !air.async.token
        air.herd @stage affinity [%0]  tile (%arg11, %arg12) in (%arg13=%c4, %arg14=%c4) args(%arg15=%alloc_1, %arg16=%alloc_3) : memref<32xf32, 1>, memref<4x4x8xf32, 1> attributes {x_loc = 0 : i64, y_loc = 0 : i64} {
          %c0_5 = arith.constant 0 : index
          %c1_6 = arith.constant 1 : index
          %c8_7 = arith.constant 8 : index
          %alloc_8 = memref.alloc() : memref<8xf32, 2>
          %1 = affine.apply #map4(%arg11)
          affine.if #set(%arg11, %arg12) {
            air.dma_memcpy_nd (%alloc_8[] [] [], %arg15[%1] [%c8_7] [%c1_6]) : (memref<8xf32, 2>, memref<32xf32, 1>)
            air.channel.put  @bx[%arg11, %c0_5] (%alloc_8[] [] []) : (memref<8xf32, 2>)
          } else {
            air.channel.get  @bx[%arg11, %arg12] (%alloc_8[] [] []) : (memref<8xf32, 2>)
          }
          air.dma_memcpy_nd (%arg16[%arg11, %arg12, %c0_5] [%c1_6, %c1_6, %c8_7] [%c8_7, %c8_7, %c1_6], %alloc_8[] [] []) : (memref<4x4x8xf32, 1>, memref<8xf32, 2>)
          memref.dealloc %alloc_8 : memref<8xf32, 2>
        }
        air.herd @compute affinity [%0]  tile (%arg11, %arg12) in (%arg13=%c4, %arg14=%c4) args(%arg15=%alloc, %arg16=%alloc_3, %arg17=%alloc_4) : memref<32x32xf32, 1>, memref<4x4x8xf32, 1>, memref<4x4x8xf32, 1> attributes {x_loc = 0 : i64, y_loc = 0 : i64} {
          %c0_5 = arith.constant 0 : index
          %c1_6 = arith.constant 1 : index
          %c8_7 = arith.constant 8 : index
          %c32_8 = arith.constant 32 : index
          %cst = arith.constant 0.000000e+00 : f32
          %alloc_9 = memref.alloc() : memref<8x8xf32, 2>
          %alloc_10 = memref.alloc() : memref<8xf32, 2>
          %alloc_11 = memref.alloc() : memref<8xf32, 2>
          %1 = affine.apply #map4(%arg12)
          %2 = affine.apply #map4(%arg11)
          air.dma_memcpy_nd (%alloc_9[] [] [], %arg15[%1, %2] [%c8_7, %c8_7] [%c32_8, %c1_6]) : (memref<8x8xf32, 2>, memref<32x32xf32, 1>)
          air.dma_memcpy_nd (%alloc_10[] [] [], %arg16[%arg11, %arg12, %c0_5] [%c1_6, %c1_6, %c8_7] [%c8_7, %c8_7, %c1_6]) : (memref<8xf32, 2>, memref<4x4x8xf32, 1>)
          affine.for %arg18 = 0 to 8 {
            affine.store %cst, %alloc_11[%arg18] : memref<8xf32, 2>
          }
          affine.for %arg18 = 0 to 8 {
            affine.for %arg19 = 0 to 8 {
              %3 = affine.load %alloc_9[%arg18, %arg19] : memref<8x8xf32, 2>
              %4 = affine.load %alloc_10[%arg19] : memref<8xf32, 2>
              %5 = affine.load %alloc_11[%arg18] : memref<8xf32, 2>
              %6 = arith.mulf %3, %4 : f32
              %7 = arith.addf %5, %6 : f32
              affine.store %7, %alloc_11[%arg18] : memref<8xf32, 2>
            }
          }
          air.dma_memcpy_nd (%arg17[%arg11, %arg12, %c0_5] [%c1_6, %c1_6, %c8_7] [%c8_7, %c8_7, %c1_6], %alloc_11[] [] []) : (memref<4x4x8xf32, 1>, memref<8xf32, 2>)
          memref.dealloc %alloc_9 : memref<8x8xf32, 2>
          memref.dealloc %alloc_10 : memref<8xf32, 2>
          memref.dealloc %alloc_11 : memref<8xf32, 2>
        }
        air.herd @reduce affinity [%0]  tile (%arg11, %arg12) in (%arg13=%c4, %arg14=%c4) args(%arg15=%alloc_4, %arg16=%alloc_2) : memref<4x4x8xf32, 1>, memref<32xf32, 1> attributes {x_loc = 0 : i64, y_loc = 0 : i64} {
          %c0_5 = arith.constant 0 : index
          %c1_6 = arith.constant 1 : index
          %c8_7 = arith.constant 8 : index
          %alloc_8 = memref.alloc() : memref<8xf32, 2>
          %alloc_9 = memref.alloc() : memref<8xf32, 2>
          air.dma_memcpy_nd (%alloc_8[] [] [], %arg15[%arg11, %arg12, %c0_5] [%c1_6, %c1_6, %c8_7] [%c8_7, %c8_7, %c1_6]) : (memref<8xf32, 2>, memref<4x4x8xf32, 1>)
          %1 = affine.apply #map5(%arg11)
          %2 = affine.apply #map4(%arg12)
          affine.if #set1(%arg11, %arg12) {
            air.channel.put  @red[%1, %arg12] (%alloc_8[] [] []) : (memref<8xf32, 2>)
          } else {
            air.channel.get  @red[%arg11, %arg12] (%alloc_9[] [] []) : (memref<8xf32, 2>)
            affine.for %arg17 = 0 to 8 {
              %3 = affine.load %alloc_8[%arg17] : memref<8xf32, 2>
              %4 = affine.load %alloc_9[%arg17] : memref<8xf32, 2>
              %5 = arith.addf %3, %4 : f32
              affine.store %5, %alloc_8[%arg17] : memref<8xf32, 2>
            }
            affine.if #set2(%arg11, %arg12) {
              air.dma_memcpy_nd (%arg16[%2] [%c8_7] [%c1_6], %alloc_8[] [] []) : (memref<32xf32, 1>, memref<8xf32, 2>)
            } else {
              air.channel.put  @red[%1, %arg12] (%alloc_8[] [] []) : (memref<8xf32, 2>)
            }
          }
          memref.dealloc %alloc_8 : memref<8xf32, 2>
          memref.dealloc %alloc_9 : memref<8xf32, 2>
        }
        air.dma_memcpy_nd (%arg10[] [] [], %alloc_2[] [] []) : (memref<32xf32>, memref<32xf32, 1>)
        memref.dealloc %alloc : memref<32x32xf32, 1>
        memref.dealloc %alloc_1 : memref<32xf32, 1>
        memref.dealloc %alloc_2 : memref<32xf32, 1>
        memref.dealloc %alloc_3 : memref<4x4x8xf32, 1>
        memref.dealloc %alloc_4 : memref<4x4x8xf32, 1>
      }
    }
    return
  }
}

