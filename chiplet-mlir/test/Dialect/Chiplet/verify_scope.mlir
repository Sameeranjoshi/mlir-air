// RUN: chiplet-opt %s -verify-diagnostics -split-input-file | FileCheck %s

// Stray partition_id outside a launch should fail verification.
func.func @bad_partition_id() {
  // expected-error @below {{must appear lexically inside a 'chiplet.launch' region}}
  %pid = chiplet.partition_id : index
  return
}

// -----

// Stray worker_id outside a task should fail verification.
func.func @bad_worker_id() {
  // expected-error @below {{must appear lexically inside a 'chiplet.task' region}}
  %wid = chiplet.worker_id : index
  return
}

// -----

// A worker_id nested correctly should pass. Use CHECK to assert round-trip.
// CHECK-LABEL: func.func @good_worker_id
func.func @good_worker_id() {
  chiplet.launch num_chiplets = 8 {
    chiplet.task level = <chiplet> {
      // CHECK: chiplet.worker_id : index
      %wid = chiplet.worker_id : index
    }
  }
  return
}
