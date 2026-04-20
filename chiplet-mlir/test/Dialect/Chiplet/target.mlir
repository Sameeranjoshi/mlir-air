// RUN: chiplet-opt %s | chiplet-opt | FileCheck %s

// CHECK-LABEL: module
// CHECK-SAME: chiplet.target = #chiplet.target<num_chiplets = 8, workers_per_chiplet = 31, l2_capacity_bytes = 4194304>
module attributes {
  chiplet.target = #chiplet.target<num_chiplets = 8, workers_per_chiplet = 31, l2_capacity_bytes = 4194304>
} {}
