// RUN: chiplet-opt %s | chiplet-opt | FileCheck %s

// CHECK-LABEL: func.func @launch_empty
func.func @launch_empty() {
  // CHECK: chiplet.launch num_chiplets = 8
  chiplet.launch num_chiplets = 8 {
  }
  return
}

// CHECK-LABEL: func.func @launch_with_partition_id
func.func @launch_with_partition_id() {
  // CHECK: chiplet.launch num_chiplets = 4
  chiplet.launch num_chiplets = 4 {
    // CHECK: %{{.*}} = chiplet.partition_id : index
    %pid = chiplet.partition_id : index
  }
  return
}
