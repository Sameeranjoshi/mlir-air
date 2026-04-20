// RUN: chiplet-opt %s | chiplet-opt | FileCheck %s

// CHECK-LABEL: func.func @task_scopes
func.func @task_scopes() {
  chiplet.launch num_chiplets = 8 {
    // CHECK: chiplet.task level = <chiplet>
    chiplet.task level = <chiplet> {
      // CHECK: chiplet.task level = <cu>
      chiplet.task level = <cu> {
        // CHECK: %{{.*}} = chiplet.worker_id : index
        %wid = chiplet.worker_id : index
      }
      // CHECK: chiplet.task level = <wavefront>
      chiplet.task level = <wavefront> {
      }
    }
    // CHECK: chiplet.task level = <device>
    chiplet.task level = <device> {
    }
  }
  return
}
