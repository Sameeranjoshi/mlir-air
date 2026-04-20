// RUN: chiplet-opt %s | chiplet-opt | FileCheck %s

// Phase 1 acceptance test: a hand-written IR expressing Fleet's gate_up+SiLU
// Chiplet-task at bs=1 on MI350. Structure mirrors Figure 4a (Fleet):
//   - one chiplet.launch with num_chiplets = 8
//   - one chiplet.task level = chiplet representing the 8 per-XCD tasks
//   - a fused SiLU wavefront task inside
//   - an event.wait upstream edge and an event.signal downstream edge
//   - weight streaming cache modifier (Fleet §4.1 three-tier policy)
//   - a chiplet-scope fence before the signal (L2-local coordination, §5.2)
//
// No inner MFMA body is generated in the MVP — that's Sub-project D's concern.
// This file must round-trip identically; it's the gate on Phase 1 acceptance.

module attributes {
  chiplet.target = #chiplet.target<num_chiplets = 8, workers_per_chiplet = 31, l2_capacity_bytes = 4194304>
} {

// CHECK-LABEL: func.func @gate_up_silu
func.func @gate_up_silu(
    %w_gu   : tensor<4096x24576xbf16, #chiplet.scope<device>>,
    %ev_in  : !chiplet.event,
    %ev_out : !chiplet.event) {

  // CHECK: chiplet.launch num_chiplets = 8
  chiplet.launch num_chiplets = 8 {

    // CHECK: %{{.*}} = chiplet.partition_id : index
    %pid = chiplet.partition_id : index

    // CHECK: chiplet.task level = <chiplet>
    chiplet.task level = <chiplet> {

      // Wait on upstream RMSNorm completion (Figure 4a).
      // CHECK: chiplet.event.wait %{{.*}} : !chiplet.event
      chiplet.event.wait %ev_in : !chiplet.event

      // Load this XCD's weight partition into L2 with streaming policy
      // (Fleet §4.1: sc1=1, nt=1 — short-lived reuse, no LRU displacement).
      // Whole-tensor copy in MVP; strided slicing is Phase 2's concern.
      // CHECK: %{{.*}} = chiplet.copy %{{.*}} {modifier = #chiplet.cache<streaming>}
      %w_slice = chiplet.copy %w_gu {modifier = #chiplet.cache<streaming>}
        : tensor<4096x24576xbf16, #chiplet.scope<device>>
       -> tensor<4096x3072xbf16, #chiplet.scope<chiplet>>

      // MFMA inner loop — opaque in the MVP. A worker_id use stands in
      // for per-worker tile iteration in the SPMD body.
      // CHECK: %{{.*}} = chiplet.worker_id : index
      %wid = chiplet.worker_id : index

      // Fused SiLU (wavefront task), Fleet §4.1 — eliminates the separate
      // SiLU Chiplet-task and its output buffer, raising L2 hit rate.
      // CHECK: chiplet.task level = <wavefront>
      chiplet.task level = <wavefront> {
        // SiLU element-wise body — opaque in MVP.
      }

      // Intra-XCD coordination before cross-XCD signalling (Figure 5).
      // CHECK: chiplet.fence scope = <chiplet>
      chiplet.fence scope = <chiplet>
    }

    // Cross-XCD signalling edge to the downstream Chiplet-task (down_proj).
    // CHECK: chiplet.event.signal %{{.*}} : !chiplet.event
    chiplet.event.signal %ev_out : !chiplet.event
  }

  return
}

}
