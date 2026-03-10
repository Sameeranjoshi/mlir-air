# AIRDependency Pass: Implementation Reference Guide

A quick technical reference for understanding the C++ implementation.

---

## High-Level Flow

```
Module
  ↓
For each Function:
  ↓
[PHASE 1] Walk all operations:
  - Identify sync ops that need async wrapping
  - Create air.execute for computation ops
  - Create async variants for memory/channel ops
  - Add empty dependency lists

[PHASE 2] Trace dependencies:
  - Analyze data flow (read-write relationships)
  - Build ExecuteGraph (directed adjacency map)
  - Connect ops that have dependencies
  - Mark which operations depend on which

[PHASE 3] Apply dependencies:
  - Add [%token1, %token2, ...] to each async op
  - Insert air.wait_all ops where needed
  - Update affine.yield / scf.yield to return tokens

Result: ACDG (Async ops with explicit tokens)
```

---

## Key Data Structures

### 1. ExecuteNode (Graph Vertex)
```cpp
struct executeNode {
  std::string asyncEventName;    // "execute_0", "dma_1", etc
  std::string asyncEventType;    // "execute", "dma", "channel", "herd", etc
  std::string color;             // For graph visualization
  std::string shape;             // For graph visualization
  std::string style;             // For graph visualization
  unsigned operationId;          // Unique ID for the operation
};
```

### 2. ExecuteGraph (Graph Structure)
```cpp
using ExecuteGraph = air::TypedDirectedAdjacencyMap<executeNode>;

// Methods:
// - addVertex(nodeData) → VertexId
// - addEdge(from_id, to_id) → creates dependency
// - getChildren(vertex_id) → gets dependent ops
// - getParents(vertex_id) → gets producer ops
```

### 3. Operation ID Tracking
```cpp
unsigned ExecuteOpID = 0;        // For air.execute ops
unsigned HierarchyOpID = 0;      // For herd/segment/launch
unsigned WaitAllOpID = 0;        // For air.wait_all ops
unsigned ChannelOpID = 0;        // For channel operations
unsigned DmaOpID = 0;            // For DMA operations

void assignOpId(Operation *op) {
  // Assigns unique {id = N : i32} attribute
  op->setAttr("id", IntegerAttr::get(..., id_value));
}
```

---

## Phase 1: Creating Async Operations

### Pattern: Operation → Async Wrapper

#### For Synchronous Ops (memref.alloc, linalg.matmul, func.call):
```cpp
void createAsyncExecute(IRRewriter &rewriter, Operation *op, Type resultType = nullptr) {
  // 1. Create air.execute op
  auto executeOp = rewriter.create<air::ExecuteOp>(
      op->getLoc(),
      resultType ? TypeRange{AsyncTokenType, resultType} : TypeRange{AsyncTokenType},
      /*dependencies=*/SmallVector<Value>()  // EMPTY - filled in Phase 2
  );

  // 2. Move original op into execute body
  executeOp.getRegion().push_back(new Block);
  auto &block = executeOp.getRegion().front();

  // 3. Insert execute_terminator
  if (resultType) {
    // For ops with results: return the result
    rewriter.create<air::ExecuteTerminatorOp>(op->getLoc(), op->getResults());
  } else {
    // For ops without results: return nothing
    rewriter.create<air::ExecuteTerminatorOp>(op->getLoc(), SmallVector<Value>());
  }

  // 4. Replace original op with execute op
  rewriter.replaceOp(op, executeOp->getResults());

  // 5. Register in graph
  updateAsyncExecuteGraphWithNewNode(executeOp, asyncExecuteGraph);
}
```

#### For DMA Ops (air.dma_memcpy_nd):
```cpp
void createAsyncDMA(IRRewriter &rewriter, Operation *op) {
  auto dmaOp = dyn_cast<air::DmaMemcpyNdOp>(op);

  // Create async version with empty deps
  auto asyncDmaOp = rewriter.create<air::DmaMemcpyNdOp>(
      dmaOp->getLoc(),
      dmaOp.getDmaMemcpyTy(),
      AsyncTokenType,
      SmallVector<Value>(),  // EMPTY - dependencies
      dmaOp.getOperands()    // Same operands as original
  );

  rewriter.replaceOp(dmaOp, asyncDmaOp->getResults());
  updateAsyncExecuteGraphWithNewNode(asyncDmaOp, asyncExecuteGraph);
}
```

#### For Channel Ops (air.channel.put/get):
```cpp
void createAsyncChannel(IRRewriter &rewriter, Operation *op) {
  auto chanOp = dyn_cast<air::ChannelInterface>(op);

  // Get the channel operation's async variant
  // channel.put → channel.put async [...]
  // channel.get → channel.get async [...]

  auto asyncChanOp = air::createAsyncChannelOp(
      rewriter,
      op->getLoc(),
      SmallVector<Value>(),  // EMPTY - dependencies
      chanOp
  );

  rewriter.replaceOp(chanOp, asyncChanOp->getResults());
  updateAsyncExecuteGraphWithNewNode(asyncChanOp, asyncExecuteGraph);
}
```

#### For Hierarchy Ops (air.herd, air.segment, air.launch):
```cpp
void createAsyncHierarchyImpls(IRRewriter &rewriter, air::HierarchyInterface hierOp) {
  // 1. Mark the hierarchy op as async
  hierOp->setAttr("async", UnitAttr::get(rewriter.getContext()));

  // 2. Change return type to include AsyncTokenType
  // Before: herd @h0 tile(...) { ... } (no return)
  // After:  herd @h0 async tile(...) { ... } (returns token)

  // 3. Recursively transform the body
  for (auto &region : hierOp->getRegions()) {
    runOnRegion(&region, rewriter);  // Recursive call!
  }

  assignOpId(hierOp);
  updateAsyncExecuteGraphWithNewNode(hierOp, asyncExecuteGraph);
}
```

---

## Phase 2: Dependency Analysis

### Pattern: Find Producers for Each Op

#### Core Algorithm:
```cpp
// For each operation in the graph:
for (auto op2 : module.getOps()) {
  if (!isAsyncOp(op2) && !isa<WaitAllOp>(op2)) continue;

  // Step 1: Find operands (what it reads)
  SmallVector<Value> operands = getMemRefOperands(op2);

  for (auto operand : operands) {
    // Step 2: Find producer (what writes this value)
    Operation *producerOp = operand.getDefiningOp();

    // Step 3: Create dependency edge
    // This op2 depends on producerOp
    auto vertex_producer = getOrCreateGraphVertex(producerOp);
    auto vertex_consumer = getOrCreateGraphVertex(op2);
    asyncExecuteGraph.addEdge(vertex_producer, vertex_consumer);
  }
}
```

#### Handling Different Operation Types:

**For ExecuteOp (wraps computation):**
```cpp
if (auto execOp = dyn_cast<air::ExecuteOp>(op)) {
  // Get operations inside the execute region
  SmallVector<Value> deps;

  // The execute's operands are the values it depends on
  for (auto operand : execOp.getOperands()) {
    auto producerOp = operand.getDefiningOp();
    if (producerOp && isAsyncOp(producerOp)) {
      deps.push_back(producerOp->getResult(0));  // async token
    }
  }

  // Store for Phase 3
  opToDependenciesMap[execOp] = deps;
}
```

**For DmaOp (reads from source, writes to destination):**
```cpp
if (auto dmaOp = dyn_cast<air::DmaMemcpyNdOp>(op)) {
  auto srcMemRef = dmaOp.getSrc();      // Read dependency
  auto dstMemRef = dmaOp.getDst();      // Write dependency

  SmallVector<Value> deps;

  // Producer of source memref
  if (auto srcProducer = srcMemRef.getDefiningOp()) {
    if (isAsyncOp(srcProducer)) {
      deps.push_back(srcProducer->getResult(0));
    }
  }

  // Previous writer to destination memref (WAW dependency)
  if (auto dstProducer = dstMemRef.getDefiningOp()) {
    if (isAsyncOp(dstProducer)) {
      deps.push_back(dstProducer->getResult(0));
    }
  }

  opToDependenciesMap[dmaOp] = deps;
}
```

**For ChannelOp (strict ordering):**
```cpp
if (auto chanOp = dyn_cast<air::ChannelInterface>(op)) {
  SmallVector<Value> deps;

  // Find all previous operations on same channel
  for (auto prevOp : allPreviousOps) {
    if (auto prevChanOp = dyn_cast<air::ChannelInterface>(prevOp)) {
      if (prevChanOp.getChannel() == chanOp.getChannel()) {
        // Operations on same channel are strictly ordered
        deps.push_back(prevChanOp->getResult(0));
        break;  // Only immediate predecessor
      }
    }
  }

  opToDependenciesMap[chanOp] = deps;
}
```

### Transitive Dependency Check (Optional Optimization)
```cpp
// After building graph, remove redundant edges
// If A→B and B→C exist, we don't need A→C (transitive)
// This is done in AIRDependencyCanonicalize pass
void removeTransitiveEdges(ExecuteGraph &graph) {
  for (auto vertex : graph.getAllVertices()) {
    auto children = graph.getChildren(vertex);
    for (auto child : children) {
      auto grandchildren = graph.getChildren(child);
      // If A→B→C, remove A→C
      auto it = std::find(children.begin(), children.end(), grandchildren);
      if (it != children.end()) {
        graph.removeEdge(vertex, *it);
      }
    }
  }
}
```

---

## Phase 3: Applying Dependencies

### Pattern: Add Tokens to Operations

#### For Regular Async Ops:
```cpp
void applyDependencies(Operation *op, SmallVector<Value> &deps) {
  if (auto execOp = dyn_cast<air::ExecuteOp>(op)) {
    // Replace operand list with dependencies
    SmallVector<Value> newOperands = deps;
    execOp->setOperands(newOperands);
  } else if (auto dmaOp = dyn_cast<air::DmaMemcpyNdOp>(op)) {
    // DMA has specific dependency slots
    dmaOp.setAsyncDependencies(deps);
  } else if (auto chanOp = dyn_cast<air::ChannelInterface>(op)) {
    // Channel has async dependencies
    chanOp.setAsyncDependencies(deps);
  }
}
```

#### For Loop Bodies (scf.for, affine.for):
```cpp
void transformLoop(Operation *loopOp, IRRewriter &rewriter) {
  // 1. Change return type to include AsyncTokenType
  SmallVector<Type> oldTypes = loopOp.getResultTypes();
  SmallVector<Type> newTypes(oldTypes);
  newTypes.push_back(AsyncTokenType);
  loopOp->getOperand(0).setType(AsyncTokenType);  // iter_args

  // 2. Update body to return token
  auto &body = loopOp.getBody();
  for (auto &op : body->getOps()) {
    if (auto yieldOp = dyn_cast<scf::YieldOp>(&op)) {
      // scf.yield %result → scf.yield %result, %token
      auto finalToken = body->back().getOperand(0);  // Last op's token
      yieldOp.getOperandsMutable().append(finalToken);
    }
  }

  // 3. Update loop to carry token through iterations
  // Each iteration starts with previous iteration's token
}
```

#### For Branches (scf.if, affine.if):
```cpp
void transformBranch(Operation *ifOp, IRRewriter &rewriter) {
  // 1. Change result type to AsyncTokenType
  // scf.if %cond { ... } else { ... }
  // →
  // scf.if %cond -> !air.async.token { ... } else { ... }

  SmallVector<Type> resultTypes = {AsyncTokenType};
  ifOp->setOperand(ifOp->getNumOperands(), resultTypes);

  // 2. Update each branch to yield token
  // then-branch: affine.yield %result
  // →
  // then-branch: %token = air.wait_all async [...]
  //              affine.yield %token

  // Find yield ops in each branch
  auto walkBranches = [&](Operation *branch) {
    branch->walk([&](Operation *op) {
      if (auto yieldOp = dyn_cast<affine::AffineYieldOp>(op)) {
        auto branchToken = /* last token in branch */;
        // Replace: affine.yield operands
        //     with: affine.yield branchToken
        yieldOp.getOperandsMutable().clear();
        yieldOp.getOperandsMutable().append(branchToken);
      }
    });
  };

  // Apply to both branches
  // Similar for scf.if
}
```

#### For Memory Deallocation:
```cpp
void transformDealloc(memref::DeallocOp deallocOp, IRRewriter &rewriter) {
  // Find all operations that use the deallocated memref
  auto memrefToDealloc = deallocOp.getMemref();
  SmallVector<Value> producerTokens;

  // Collect tokens from all producers of this memref
  for (auto *user : memrefToDealloc.getUsers()) {
    if (user != deallocOp) {
      if (isAsyncOp(user)) {
        producerTokens.push_back(user->getResult(0));
      }
    }
  }

  // Wrap dealloc in execute that depends on all producers
  auto deallocExecute = rewriter.create<air::ExecuteOp>(
      deallocOp.getLoc(),
      AsyncTokenType,
      producerTokens  // IMPORTANT: must wait for all uses
  );

  // Move dealloc into the execute
  // This ensures memory isn't freed until all ops are done
}
```

---

## Utility Functions

### Check if Operation is Async:
```cpp
bool isAsyncOp(Operation *op) {
  return isa<air::ExecuteOp, air::DmaMemcpyNdOp>(op) ||
         isa<air::ChannelInterface>(op) ||
         (isa<air::HierarchyInterface>(op) &&
          op->hasAttr("async"));
}
```

### Get Async Token (Result 0):
```cpp
Value getAsyncToken(Operation *op) {
  if (auto execOp = dyn_cast<air::ExecuteOp>(op)) {
    return execOp.getAsyncToken();  // %token
  } else if (auto dmaOp = dyn_cast<air::DmaMemcpyNdOp>(op)) {
    return dmaOp->getResult(0);     // %token
  }
  // ... similar for others
}
```

### Get Memref Operands:
```cpp
SmallVector<Value> getMemRefOperands(Operation *op) {
  SmallVector<Value> memrefs;
  for (auto operand : op->getOperands()) {
    if (llvm::isa<BaseMemRefType>(operand.getType())) {
      memrefs.push_back(operand);
    }
  }
  return memrefs;
}
```

### Create air.wait_all:
```cpp
Value createWaitAllOp(IRRewriter &rewriter, Location loc,
                      SmallVector<Value> dependencies) {
  auto waitAllOp = rewriter.create<air::WaitAllOp>(
      loc,
      AsyncTokenType,
      dependencies  // [%t1, %t2, %t3]
  );
  assignOpId(waitAllOp);
  return waitAllOp->getResult(0);
}
```

---

## Debugging Functions

### Print Graph Structure:
```cpp
void dumpExecuteGraph(ExecuteGraph &graph) {
  for (auto vertex : graph.getAllVertices()) {
    auto nodeData = graph.getNode(vertex);
    llvm::errs() << "Vertex " << vertex << ": " << nodeData.asyncEventName << "\n";

    for (auto child : graph.getChildren(vertex)) {
      auto childData = graph.getNode(child);
      llvm::errs() << "  → " << childData.asyncEventName << "\n";
    }
  }
}
```

### Verify Async Completeness:
```cpp
bool verifyAllOpsAreAsync(Operation *op) {
  bool valid = true;
  op->walk([&](Operation *nestedOp) {
    // Skip terminators, constants, etc.
    if (nestedOp->mightHaveTrait<OpTrait::IsTerminator>()) return;
    if (isa<arith::ConstantOp>(nestedOp)) return;

    // Check if operation is async or has no results
    if (!isAsyncOp(nestedOp) && nestedOp->getNumResults() > 0) {
      llvm::errs() << "WARNING: Non-async op with results: " << *nestedOp << "\n";
      valid = false;
    }
  });
  return valid;
}
```

---

## Common Patterns in Code

### Pattern: Register Operation in Graph
```cpp
void updateAsyncExecuteGraphWithNewNode(Operation *op, ExecuteGraph &graph) {
  executeNode node;
  node.asyncEventName = op->getName().getStringRef().str();
  node.operationId = ExecuteOpID++;

  auto vertexId = graph.addVertex(node);
  opToVertexMap[op] = vertexId;
}
```

### Pattern: Collect Dependencies from Region
```cpp
SmallVector<Value> collectRegionDependencies(Region &region) {
  SetVector<Value> deps;

  region.walk([&](Operation *op) {
    for (auto operand : op->getOperands()) {
      if (auto asyncProducerOp = dyn_cast<Operation>(operand.getDefiningOp())) {
        if (isAsyncOp(asyncProducerOp)) {
          deps.insert(getAsyncToken(asyncProducerOp));
        }
      }
    }
  });

  return deps.takeVector();
}
```

### Pattern: Recursive Region Transformation
```cpp
void runOnRegion(Region *region, IRRewriter &rewriter) {
  for (auto &block : region->getBlocks()) {
    // Process each operation in the block
    for (auto op : llvm::make_early_inc_range(block.getOps())) {
      if (isa<air::HierarchyInterface>(op)) {
        // Recursively transform nested hierarchy
        createAsyncHierarchyImpls(rewriter, cast<air::HierarchyInterface>(op));
      } else {
        // Transform regular op
        // ...
      }
    }
  }
}
```

---

## Test Case Mapping

| Test File | Pattern Tested |
|-----------|----------------|
| `dma_memcpy_nd.mlir` | Basic DMA async wrapping |
| `matmul_nd.mlir` | Producer→consumer chains |
| `affine_if.mlir` | Branch token yielding |
| `scf_for.mlir` | Loop iteration threading |
| `scf_if.mlir` | Conditional token handling |
| `air_channel.mlir` | Channel operation ordering |
| `air_hierarchy.mlir` | Nested herd transformation |
| `parallel_herds.mlir` | Independent parallelization |
| `reshape_dependency.mlir` | Deallocation synchronization |

---

## Key Takeaways

1. **Phase 1**: Wrapping synchronous ops → async ops with empty dependency lists
2. **Phase 2**: Analyzing data flow → building a dependency graph
3. **Phase 3**: Applying dependencies → adding tokens to async ops

The pass creates an **explicit ACDG** where every operation knows exactly what it depends on, enabling optimal scheduling and parallelization.

