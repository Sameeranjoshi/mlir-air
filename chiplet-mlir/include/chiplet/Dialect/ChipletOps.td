//===- ChipletOps.td - Chiplet ops -------------------------*- tablegen -*-===//
#ifndef CHIPLET_OPS
#define CHIPLET_OPS

include "chiplet/Dialect/ChipletDialect.td"
include "chiplet/Dialect/ChipletAttrs.td"
include "chiplet/Dialect/ChipletTypes.td"
include "mlir/IR/BuiltinAttributes.td"
include "mlir/IR/BuiltinTypes.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

//===----------------------------------------------------------------------===//
// chiplet.launch
//===----------------------------------------------------------------------===//

def Chiplet_LaunchOp : Chiplet_Op<"launch", [
    SingleBlock, NoTerminator, NoRegionArguments
  ]> {
  let summary = "Launch a chiplet task graph over num_chiplets partitions.";
  let description = [{
    Top-level op binding a chiplet task graph to the enclosing module's
    target. `num_chiplets` declares the number of logical partitions (i.e. the
    range of `chiplet.partition_id`). The body contains `chiplet.task` ops
    and may use `chiplet.partition_id` as a symbolic index.

    In the MVP, `chiplet.launch` has no operands and no results — tensor
    plumbing through the region is handled by ordinary SSA capture.

    Example:
    ```mlir
    chiplet.launch num_chiplets = 8 {
      %pid = chiplet.partition_id : index
      chiplet.task { level = #chiplet.scope<chiplet> } {
        // ...
      }
    }
    ```
  }];

  let arguments = (ins I64Attr:$num_chiplets);
  let results = (outs);
  let regions = (region SizedRegion<1>:$body);

  let assemblyFormat = [{
    `num_chiplets` `=` $num_chiplets attr-dict-with-keyword $body
  }];
}

//===----------------------------------------------------------------------===//
// chiplet.task
//===----------------------------------------------------------------------===//

def Chiplet_TaskOp : Chiplet_Op<"task", [
    SingleBlock, NoTerminator, NoRegionArguments
  ]> {
  let summary = "A task at a named scope (wavefront / cu / chiplet / device).";
  let description = [{
    Describes a unit of work at `level`. Body is SPMD across workers at that
    level; `chiplet.worker_id` gives the worker rank within the current task.
    Nesting rules (e.g. a cu-task cannot contain a chiplet-task) are not
    enforced in the MVP — see docs/design_decisions.md.

    Example:
    ```mlir
    chiplet.task { level = #chiplet.scope<chiplet> } {
      %wid = chiplet.worker_id : index
      // ...
    }
    ```
  }];

  let arguments = (ins Chiplet_ScopeAttr:$level);
  let results = (outs);
  let regions = (region SizedRegion<1>:$body);

  let assemblyFormat = [{
    `level` `=` $level attr-dict-with-keyword $body
  }];
}

//===----------------------------------------------------------------------===//
// chiplet.partition_id
//===----------------------------------------------------------------------===//

def Chiplet_PartitionIdOp : Chiplet_Op<"partition_id", [Pure]> {
  let summary = "Symbolic partition index inside a chiplet.launch body.";
  let description = [{
    Returns the partition index `[0, num_chiplets)` at which the enclosing
    `chiplet.launch` invocation executes. Valid only lexically inside a
    `chiplet.launch` region.
  }];

  let arguments = (ins);
  let results = (outs Index:$result);

  let assemblyFormat = "attr-dict `:` type($result)";
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// chiplet.worker_id
//===----------------------------------------------------------------------===//

def Chiplet_WorkerIdOp : Chiplet_Op<"worker_id", [Pure]> {
  let summary = "Worker rank within the enclosing chiplet.task body.";
  let description = [{
    Returns the worker index in the SPMD body of the enclosing
    `chiplet.task`. Valid only lexically inside a `chiplet.task` region.
  }];

  let arguments = (ins);
  let results = (outs Index:$result);

  let assemblyFormat = "attr-dict `:` type($result)";
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// chiplet.copy
//===----------------------------------------------------------------------===//

def Chiplet_CopyOp : Chiplet_Op<"copy", []> {
  let summary = "Data movement between scopes with a cache-modifier policy.";
  let description = [{
    Moves a tensor from one scope (encoded in the input's tensor encoding) to
    another (encoded in the result). The `modifier` attribute selects Fleet's
    three-tier cache policy (Fleet §4.1): `cache_all` (default LRU),
    `streaming` (sc1=1, nt=1 — weight loads), or `non_temporal`
    (nt=1 — activation stores, cross-XCD polling).

    Strided-slice offsets and sizes are deferred to a later sub-project; this
    MVP form moves whole tensors.

    Example:
    ```mlir
    %w_slice = chiplet.copy %w
      { modifier = #chiplet.cache<streaming> }
      : tensor<4096x3072xbf16, #chiplet.scope<device>>
     -> tensor<4096x3072xbf16, #chiplet.scope<chiplet>>
    ```
  }];

  let arguments = (ins
    AnyRankedTensor:$source,
    Chiplet_CacheModifierAttr:$modifier
  );
  let results = (outs AnyRankedTensor:$result);

  let assemblyFormat = [{
    $source attr-dict `:` type($source) `->` type($result)
  }];
}

//===----------------------------------------------------------------------===//
// chiplet.fence
//===----------------------------------------------------------------------===//

def Chiplet_FenceOp : Chiplet_Op<"fence", []> {
  let summary = "Memory fence at a named scope.";
  let description = [{
    Synchronization at `scope`. At lowering time, the scope attribute selects
    which of Fleet's four sync patterns is emitted (Fleet §5.2):
    wavefront / cu → no global fence; chiplet → L2-local counter; device →
    `buffer_wbl2` + threadfence.
  }];

  let arguments = (ins Chiplet_ScopeAttr:$scope);
  let results = (outs);

  let assemblyFormat = "`scope` `=` $scope attr-dict";
}

//===----------------------------------------------------------------------===//
// chiplet.event.signal
//===----------------------------------------------------------------------===//

def Chiplet_EventSignalOp : Chiplet_Op<"event.signal", []> {
  let summary = "Signal completion of an event counter.";
  let description = [{
    Producer side of a Fleet event. Downstream tasks that `event.wait` on the
    same event observe completion. Lowering emits a `flat_atomic_add` with
    appropriate scope bits.
  }];

  let arguments = (ins Chiplet_EventType:$event);
  let results = (outs);

  let assemblyFormat = "$event attr-dict `:` type($event)";
}

//===----------------------------------------------------------------------===//
// chiplet.event.wait
//===----------------------------------------------------------------------===//

def Chiplet_EventWaitOp : Chiplet_Op<"event.wait", []> {
  let summary = "Wait for an event counter to signal.";
  let description = [{
    Consumer side of a Fleet event. Blocks until the matching producer has
    signalled. Lowering emits the scheduler-polling path described in
    Fleet Figure 5.
  }];

  let arguments = (ins Chiplet_EventType:$event);
  let results = (outs);

  let assemblyFormat = "$event attr-dict `:` type($event)";
}

#endif // CHIPLET_OPS
