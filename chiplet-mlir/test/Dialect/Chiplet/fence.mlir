// RUN: chiplet-opt %s | chiplet-opt | FileCheck %s

// CHECK-LABEL: func.func @fences
func.func @fences() {
  chiplet.launch num_chiplets = 8 {
    chiplet.task level = <chiplet> {
      // CHECK: chiplet.fence scope = <wavefront>
      chiplet.fence scope = <wavefront>
      // CHECK: chiplet.fence scope = <cu>
      chiplet.fence scope = <cu>
      // CHECK: chiplet.fence scope = <chiplet>
      chiplet.fence scope = <chiplet>
      // CHECK: chiplet.fence scope = <device>
      chiplet.fence scope = <device>
    }
  }
  return
}
