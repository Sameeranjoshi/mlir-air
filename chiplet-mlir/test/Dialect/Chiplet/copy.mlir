// RUN: chiplet-opt %s | chiplet-opt | FileCheck %s

// CHECK-LABEL: func.func @copy_streaming
func.func @copy_streaming(%w : tensor<4096x3072xbf16, #chiplet.scope<device>>)
    -> tensor<4096x3072xbf16, #chiplet.scope<chiplet>> {
  // CHECK: chiplet.copy %{{.*}} {modifier = #chiplet.cache<streaming>} : tensor<4096x3072xbf16, #chiplet.scope<device>> -> tensor<4096x3072xbf16, #chiplet.scope<chiplet>>
  %r = chiplet.copy %w {modifier = #chiplet.cache<streaming>}
    : tensor<4096x3072xbf16, #chiplet.scope<device>>
   -> tensor<4096x3072xbf16, #chiplet.scope<chiplet>>
  return %r : tensor<4096x3072xbf16, #chiplet.scope<chiplet>>
}

// CHECK-LABEL: func.func @copy_non_temporal
func.func @copy_non_temporal(%a : tensor<1x24576xbf16, #chiplet.scope<chiplet>>)
    -> tensor<1x24576xbf16, #chiplet.scope<device>> {
  // CHECK: chiplet.copy %{{.*}} {modifier = #chiplet.cache<non_temporal>}
  %r = chiplet.copy %a {modifier = #chiplet.cache<non_temporal>}
    : tensor<1x24576xbf16, #chiplet.scope<chiplet>>
   -> tensor<1x24576xbf16, #chiplet.scope<device>>
  return %r : tensor<1x24576xbf16, #chiplet.scope<device>>
}

// CHECK-LABEL: func.func @copy_cache_all
func.func @copy_cache_all(%w : tensor<16x16xf32>) -> tensor<16x16xf32> {
  // CHECK: chiplet.copy %{{.*}} {modifier = #chiplet.cache<cache_all>}
  %r = chiplet.copy %w {modifier = #chiplet.cache<cache_all>}
    : tensor<16x16xf32>
   -> tensor<16x16xf32>
  return %r : tensor<16x16xf32>
}
