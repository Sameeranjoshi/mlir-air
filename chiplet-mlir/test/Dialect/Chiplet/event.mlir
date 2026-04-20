// RUN: chiplet-opt %s | chiplet-opt | FileCheck %s

// CHECK-LABEL: func.func @signal_wait
func.func @signal_wait(%ev : !chiplet.event) {
  // CHECK: chiplet.event.wait %{{.*}} : !chiplet.event
  chiplet.event.wait %ev : !chiplet.event
  // CHECK: chiplet.event.signal %{{.*}} : !chiplet.event
  chiplet.event.signal %ev : !chiplet.event
  return
}
