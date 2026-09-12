// Cannon's matmul (examples/matmul/cannon_air.mlir) after
// `air-opt -air-place-herds-by-token="fabric-x=4 fabric-y=4"`, committed here
// as the air-to-spada lit test input (== examples/matmul/out/cannon_placed.mlir).
// See docs/SPADA_EMITTER_SPEC_V2.md.
//
// RUN: air-translate --air-to-spada %s | FileCheck %s

// Kernel signature: two readonly scatter streams, one writeonly gather
// stream, all block-sized 8x8 = 64 elements.
// CHECK: kernel @cannon<>(stream<f32, 64>[4, 4] readonly arg0_in,
// CHECK: stream<f32, 64>[4, 4] readonly arg1_in,
// CHECK: stream<f32, 64>[4, 4] writeonly arg2_out)

// Spec v2 §1: partitioned L2 buffers are declared with the block shape
// (ranked), and only the channel.get destinations (An/Bn) are flat.
// CHECK: f32[8, 8] l2_0
// CHECK: f32[8, 8] l2_1
// CHECK: f32[8, 8] l2_2
// CHECK: f32[8, 8] cannon_l1_0
// CHECK: f32[8, 8] cannon_l1_1
// CHECK: f32[8, 8] cannon_l1_2
// CHECK: f32[64] cannon_l1_3
// CHECK: f32[64] cannon_l1_4

// Spec v2 §4: ring classification of the two shift channels -- one short
// (parity) pair plus a wrap stream for the one-tile-wide wraparound edge.
// CHECK: stream<f32> shA_even = relative_stream(-1, 0) { hops = [(-1, 0)], channel = 0 }
// CHECK: stream<f32> shA_odd = relative_stream(-1, 0) { hops = [(-1, 0)], channel = 1 }
// CHECK: stream<f32> shA_wrap = relative_stream(3, 0) { hops = [(1, 0), (1, 0), (1, 0)], channel = 2 }
// CHECK: stream<f32> shB_even = relative_stream(0, -1) { hops = [(0, -1)], channel = 3 }
// CHECK: stream<f32> shB_odd = relative_stream(0, -1) { hops = [(0, -1)], channel = 4 }
// CHECK: stream<f32> shB_wrap = relative_stream(0, 3) { hops = [(0, 1), (0, 1), (0, 1)], channel = 5 }

// A representative single-tile compute block (tile (0, 0): x = 0 wraps on
// shA, sends west/odd on shB) with the 2-D matmul statement and the
// asynchronous-put/wait_all -> completion/await lowering (spec v2 §2)
// inside the step loop (spec v2 §3).
// CHECK: compute i16 x, i16 y in [0, 0] {
// CHECK: for i16 k in [0:4] {
// CHECK: for i16 l in [0:8] {
// CHECK: for i16 m in [0:8] {
// CHECK: for i16 n in [0:8] {
// CHECK: cannon_l1_2[l, m] = (cannon_l1_2[l, m] + (cannon_l1_0[l, n] * cannon_l1_1[n, m]))
// CHECK: }
// CHECK: }
// CHECK: }
// CHECK: completion c0 = send(cannon_l1_0, shA_wrap)
// CHECK: completion c1 = send(cannon_l1_1, shB_wrap)
// CHECK: await receive(cannon_l1_3, shA_odd)
// CHECK: await receive(cannon_l1_4, shB_odd)
// CHECK: await c0
// CHECK: await c1
// CHECK: cannon_l1_0[l, m] = cannon_l1_3[l * 8 + m]
// CHECK: cannon_l1_1[l, m] = cannon_l1_4[l * 8 + m]

#map = affine_map<(d0, d1) -> ((d1 - d0) mod 4, d0)>
#map1 = affine_map<(d0, d1) -> (d1, (d0 - d1) mod 4)>
#map2 = affine_map<(d0, d1) -> (d1, d0)>
#map3 = affine_map<(d0) -> (d0 * 8)>
#map4 = affine_map<(d0, d1) -> (((d0 + d1) mod 4) * 8)>
#map5 = affine_map<(d0) -> ((d0 - 1) mod 4)>
module {
  air.channel @shA [4, 4]
  air.channel @shB [4, 4]
  func.func @cannon(%arg0: memref<32x32xf32>, %arg1: memref<32x32xf32>, %arg2: memref<32x32xf32>) {
    %c1 = arith.constant 1 : index
    air.launch (%arg3) in (%arg4=%c1) args(%arg5=%arg0, %arg6=%arg1, %arg7=%arg2) : memref<32x32xf32>, memref<32x32xf32>, memref<32x32xf32> {
      air.segment @cannon_seg  args(%arg8=%arg5, %arg9=%arg6, %arg10=%arg7) : memref<32x32xf32>, memref<32x32xf32>, memref<32x32xf32> {
        %c4 = arith.constant 4 : index
        %alloc = memref.alloc() {air.partition = #air.partition<block = [8, 8], owner = #map>} : memref<32x32xf32, 1>
        %alloc_0 = memref.alloc() {air.partition = #air.partition<block = [8, 8], owner = #map1>} : memref<32x32xf32, 1>
        %alloc_1 = memref.alloc() {air.partition = #air.partition<block = [8, 8], owner = #map2>} : memref<32x32xf32, 1>
        air.dma_memcpy_nd (%alloc[] [] [], %arg8[] [] []) : (memref<32x32xf32, 1>, memref<32x32xf32>)
        air.dma_memcpy_nd (%alloc_0[] [] [], %arg9[] [] []) : (memref<32x32xf32, 1>, memref<32x32xf32>)
        air.herd @cannon  tile (%arg11, %arg12) in (%arg13=%c4, %arg14=%c4) args(%arg15=%alloc, %arg16=%alloc_0, %arg17=%alloc_1) : memref<32x32xf32, 1>, memref<32x32xf32, 1>, memref<32x32xf32, 1> attributes {x_loc = 0 : i64, y_loc = 0 : i64} {
          %c1_2 = arith.constant 1 : index
          %c8 = arith.constant 8 : index
          %c32 = arith.constant 32 : index
          %cst = arith.constant 0.000000e+00 : f32
          %alloc_3 = memref.alloc() : memref<8x8xf32, 2>
          %alloc_4 = memref.alloc() : memref<8x8xf32, 2>
          %alloc_5 = memref.alloc() : memref<8x8xf32, 2>
          %alloc_6 = memref.alloc() : memref<8x8xf32, 2>
          %alloc_7 = memref.alloc() : memref<8x8xf32, 2>
          %0 = affine.apply #map3(%arg12)
          %1 = affine.apply #map4(%arg11, %arg12)
          %2 = affine.apply #map4(%arg11, %arg12)
          %3 = affine.apply #map3(%arg11)
          air.dma_memcpy_nd (%alloc_3[] [] [], %arg15[%0, %1] [%c8, %c8] [%c32, %c1_2]) : (memref<8x8xf32, 2>, memref<32x32xf32, 1>)
          air.dma_memcpy_nd (%alloc_4[] [] [], %arg16[%2, %3] [%c8, %c8] [%c32, %c1_2]) : (memref<8x8xf32, 2>, memref<32x32xf32, 1>)
          affine.for %arg18 = 0 to 8 {
            affine.for %arg19 = 0 to 8 {
              affine.store %cst, %alloc_5[%arg18, %arg19] : memref<8x8xf32, 2>
            }
          }
          %4 = affine.apply #map5(%arg11)
          %5 = affine.apply #map5(%arg12)
          affine.for %arg18 = 0 to 4 {
            affine.for %arg19 = 0 to 8 {
              affine.for %arg20 = 0 to 8 {
                affine.for %arg21 = 0 to 8 {
                  %10 = affine.load %alloc_3[%arg19, %arg21] : memref<8x8xf32, 2>
                  %11 = affine.load %alloc_4[%arg21, %arg20] : memref<8x8xf32, 2>
                  %12 = affine.load %alloc_5[%arg19, %arg20] : memref<8x8xf32, 2>
                  %13 = arith.mulf %10, %11 : f32
                  %14 = arith.addf %12, %13 : f32
                  affine.store %14, %alloc_5[%arg19, %arg20] : memref<8x8xf32, 2>
                }
              }
            }
            %8 = air.channel.put async  @shA[%4, %arg12] (%alloc_3[] [] []) : (memref<8x8xf32, 2>)
            %9 = air.channel.put async  @shB[%arg11, %5] (%alloc_4[] [] []) : (memref<8x8xf32, 2>)
            air.channel.get  @shA[%arg11, %arg12] (%alloc_6[] [] []) : (memref<8x8xf32, 2>)
            air.channel.get  @shB[%arg11, %arg12] (%alloc_7[] [] []) : (memref<8x8xf32, 2>)
            air.wait_all [%8, %9]
            affine.for %arg19 = 0 to 8 {
              affine.for %arg20 = 0 to 8 {
                %10 = affine.load %alloc_6[%arg19, %arg20] : memref<8x8xf32, 2>
                affine.store %10, %alloc_3[%arg19, %arg20] : memref<8x8xf32, 2>
                %11 = affine.load %alloc_7[%arg19, %arg20] : memref<8x8xf32, 2>
                affine.store %11, %alloc_4[%arg19, %arg20] : memref<8x8xf32, 2>
              }
            }
          }
          %6 = affine.apply #map3(%arg12)
          %7 = affine.apply #map3(%arg11)
          air.dma_memcpy_nd (%arg17[%6, %7] [%c8, %c8] [%c32, %c1_2], %alloc_5[] [] []) : (memref<32x32xf32, 1>, memref<8x8xf32, 2>)
          memref.dealloc %alloc_3 : memref<8x8xf32, 2>
          memref.dealloc %alloc_4 : memref<8x8xf32, 2>
          memref.dealloc %alloc_5 : memref<8x8xf32, 2>
          memref.dealloc %alloc_6 : memref<8x8xf32, 2>
          memref.dealloc %alloc_7 : memref<8x8xf32, 2>
        }
        air.dma_memcpy_nd (%arg10[] [] [], %alloc_1[] [] []) : (memref<32x32xf32>, memref<32x32xf32, 1>)
        memref.dealloc %alloc : memref<32x32xf32, 1>
        memref.dealloc %alloc_0 : memref<32x32xf32, 1>
        memref.dealloc %alloc_1 : memref<32x32xf32, 1>
      }
    }
    return
  }
}
