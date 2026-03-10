# ACDG Learning & Analysis Resources - Complete Summary

This document summarizes all available materials for understanding the AIRDependency pass and ACDG transformations.

## 📚 Educational Documentation (6 Documents)

All located in `/scratch/general/vast/u1418973/mlir-air/`

### 1. **ACDG_QUICK_REFERENCE.txt** (1 page, printable)
**Best for:** Keeping on your desk while working
- Key syntax with examples
- Common commands
- Dependency patterns
- Golden rules for token management

### 2. **ACDG_README.md** (352 lines)
**Best for:** Navigation and choosing what to read
- Overview of all materials
- Learning paths (Beginner: 35 min, Intermediate: 80 min, Advanced: 120 min)
- Quick navigation by topic
- FAQ with cross-references

### 3. **ACDG_DEPENDENCY_GUIDE.md** (530 lines)
**Best for:** Understanding the theoretical foundation
- Comprehensive explanation of ACDG concepts
- Why async tokens matter
- Three-phase transformation in detail
- Operation-by-operation transformations
- Handling loops, branches, and special cases
- Common pitfalls and solutions

### 4. **ACDG_VISUAL_EXAMPLES.md** (547 lines)
**Best for:** Seeing concrete before/after transformations
- 5 real test case examples from the codebase:
  1. matmul_nd.mlir (producer-consumer)
  2. affine_if.mlir (nested branching)
  3. scf_for.mlir (loop iteration threading)
  4. parallel_herds.mlir (independent parallelization)
  5. memref_reshape.mlir (deallocation synchronization)
- Each includes: before code, after code, dependency diagram, execution timeline
- Pattern summary with ASCII visualizations

### 5. **ACDG_IMPLEMENTATION_REFERENCE.md** (566 lines)
**Best for:** Understanding the C++ implementation
- Key data structures (ExecuteNode, ExecuteGraph)
- Phase 1: Converting to async operations
- Phase 2: Dependency analysis algorithm
- Phase 3: Applying dependencies
- Code patterns and examples
- Utility functions
- Test case mapping

### 6. **VISUALIZE_DEPENDENCIES.md** (452 lines)
**Best for:** Hands-on practical analysis
- How to run the pass with various options
- Understanding transformed code syntax
- Real example walkthrough with diagrams
- Manual dependency extraction techniques
- Python script for analyzing dependencies
- Bash templates for GraphViz visualization
- Debugging tips and validation strategies
- Test case recommendations by difficulty

## 🛠️ Practical Analysis Tools (2 Tools)

### 1. **analyze_acdg.py** (Python script, ~150 lines)
**Location:** `/scratch/general/vast/u1418973/mlir-air/analyze_acdg.py`

**What it does:**
- Extracts all async operations from transformed MLIR
- Categorizes operations by type
- Identifies root operations (no dependencies)
- Identifies join operations (multiple dependencies)
- Shows unused tokens (potential issues)

**Usage:**
```bash
./install/bin/air-opt test.mlir -air-dependency > /tmp/output.mlir
python3 analyze_acdg.py /tmp/output.mlir
```

**Output includes:**
- Total operation count
- Breakdown by operation type
- Detailed operation list with dependencies
- Dependency pattern analysis
- Token usage statistics

### 2. **QUICK_ANALYSIS_COMMANDS.md** (Reference guide)
**Location:** `/scratch/general/vast/u1418973/mlir-air/QUICK_ANALYSIS_COMMANDS.md`

**Contains:** Copy-paste commands for:
- Analyzing single test cases
- Comparing before/after transformations
- Running different pass pipelines
- Finding specific patterns
- Running batch analyses
- Quick troubleshooting

## 📖 How to Use These Resources

### For Quick Understanding (30 minutes)
1. Read **ACDG_QUICK_REFERENCE.txt** (5 min)
2. Run one test case with analyze_acdg.py (10 min)
3. Look at **ACDG_VISUAL_EXAMPLES.md** Example 1 (15 min)

### For Conceptual Understanding (60 minutes)
1. Read **ACDG_README.md** (10 min)
2. Read **ACDG_DEPENDENCY_GUIDE.md** Part 1-3 (20 min)
3. Run test cases from **QUICK_ANALYSIS_COMMANDS.md** (20 min)
4. Review **ACDG_VISUAL_EXAMPLES.md** (10 min)

### For Deep Technical Understanding (120+ minutes)
1. Complete conceptual path above (60 min)
2. Read **ACDG_IMPLEMENTATION_REFERENCE.md** (30 min)
3. Read source code: `mlir/lib/Transform/AIRDependency.cpp` (30+ min)
4. Run analyses on different test cases (30+ min)

### For Specific Topics

**"How does branching work?"**
→ ACDG_VISUAL_EXAMPLES.md Example 2 (affine_if) + ACDG_DEPENDENCY_GUIDE.md Part 6

**"How do loops get transformed?"**
→ ACDG_VISUAL_EXAMPLES.md Example 3 (scf_for) + ACDG_DEPENDENCY_GUIDE.md Part 7

**"What's a token?"**
→ ACDG_QUICK_REFERENCE.txt + ACDG_DEPENDENCY_GUIDE.md Part 1

**"How do I debug a transformation?"**
→ VISUALIZE_DEPENDENCIES.md + analyze_acdg.py + QUICK_ANALYSIS_COMMANDS.md

**"What's the C++ code doing?"**
→ ACDG_IMPLEMENTATION_REFERENCE.md + mlir/lib/Transform/AIRDependency.cpp

## 📊 Resource Overview

| Resource | Type | Length | Focus | Level |
|----------|------|--------|-------|-------|
| QUICK_REFERENCE.txt | Cheat sheet | 1 page | Syntax & patterns | All |
| ACDG_README.md | Navigation | 352 lines | Overview & index | Beginner |
| ACDG_DEPENDENCY_GUIDE.md | Tutorial | 530 lines | Concepts & theory | Beginner |
| ACDG_VISUAL_EXAMPLES.md | Examples | 547 lines | Before/after patterns | Intermediate |
| ACDG_IMPLEMENTATION_REFERENCE.md | Reference | 566 lines | C++ code walkthrough | Advanced |
| VISUALIZE_DEPENDENCIES.md | How-to | 452 lines | Practical analysis | Beginner+ |
| analyze_acdg.py | Tool | ~150 lines | Automated analysis | All |
| QUICK_ANALYSIS_COMMANDS.md | Commands | Reference | Copy-paste recipes | All |
| PRACTICAL_ANALYSIS_GUIDE.md | Guide | ~200 lines | Workflow examples | Beginner+ |

**Total:** ~3500 lines of documentation + tools + examples

## 🚀 Quick Start

### 5-Minute Demo
```bash
cd /scratch/general/vast/u1418973/mlir-air

# Run the transformation
./install/bin/air-opt mlir/test/Transform/AIRDependency/matmul_nd.mlir -air-dependency > /tmp/output.mlir

# Analyze it
python3 analyze_acdg.py /tmp/output.mlir
```

### 15-Minute Understanding
1. Print **ACDG_QUICK_REFERENCE.txt**
2. Run 3 test cases with analyze_acdg.py
3. Look at the before/after in **ACDG_VISUAL_EXAMPLES.md**

### 1-Hour Deep Dive
1. Read **ACDG_README.md** (10 min)
2. Follow "Beginner" learning path (35 min)
3. Run recommended test cases (15 min)

## 📂 File Locations

All files are in `/scratch/general/vast/u1418973/mlir-air/`:
```
├── ACDG_QUICK_REFERENCE.txt          (1 page, print-friendly)
├── ACDG_README.md                     (navigation & learning paths)
├── ACDG_DEPENDENCY_GUIDE.md           (concepts & theory)
├── ACDG_VISUAL_EXAMPLES.md            (before/after examples)
├── ACDG_IMPLEMENTATION_REFERENCE.md   (C++ code reference)
├── VISUALIZE_DEPENDENCIES.md          (hands-on practical guide)
├── PRACTICAL_ANALYSIS_GUIDE.md        (analysis workflow examples)
├── analyze_acdg.py                    (Python analysis tool)
└── QUICK_ANALYSIS_COMMANDS.md         (copy-paste command reference)
```

## 🔗 Connections to Source Code

Materials reference these key files:
- **mlir/lib/Transform/AIRDependency.cpp** - Main pass implementation
- **mlir/include/air/Transform/AIRDependency.h** - Header with data structures
- **mlir/lib/Util/Dependency.cpp** - Utility functions
- **mlir/lib/Transform/AIRDependencyCanonicalize.cpp** - Canonicalization pass
- **mlir/lib/Transform/AIRDependencyScheduleOpt.cpp** - Scheduling optimization
- **mlir/test/Transform/AIRDependency/** - Test cases referenced throughout

## 💡 Key Insights

### What ACDG Is
An asynchronous compute data-flow graph where every operation has:
1. **Input tokens** - what it depends on
2. **Output token** - when it completes
3. **Explicit ordering** - visible in the IR

### Why It Matters
- Makes dependencies explicit for hardware scheduling
- Enables parallelization by identifying independent operations
- Ensures memory safety by controlling when resources are free
- Creates an acyclic execution DAG

### The Three Phases
| Phase | Input | Output | Purpose |
|-------|-------|--------|---------|
| 1 | Sync ops | Async ops | Create token structure |
| 2 | Async ops | Dependency graph | Analyze data flow |
| 3 | Dependencies | Ops with [tokens] | Apply synchronization |

## 🎯 What You Can Do With These Materials

✅ **Understand** the AIRDependency pass conceptually
✅ **Trace** how specific operations are transformed
✅ **Analyze** dependency structures in any MLIR file
✅ **Debug** unexpected transformations
✅ **Compare** before/after to verify correctness
✅ **Extend** the pass with knowledge of its implementation
✅ **Teach** others about ACDG and asynchronous concurrency
✅ **Optimize** by understanding dependency bottlenecks

## 📞 Getting Help

### For Conceptual Questions
→ Check **ACDG_README.md** FAQ first, then **ACDG_DEPENDENCY_GUIDE.md**

### For Practical Questions
→ See **QUICK_ANALYSIS_COMMANDS.md** and **PRACTICAL_ANALYSIS_GUIDE.md**

### For Implementation Questions
→ Read **ACDG_IMPLEMENTATION_REFERENCE.md** alongside source code

### For Debugging
→ Use **analyze_acdg.py** to get immediate insights, then check **VISUALIZE_DEPENDENCIES.md**

## 🎓 Learning Philosophy

These materials teach by:
1. **Concepts first** - understand the why
2. **Examples second** - see concrete before/after
3. **Code last** - read implementation with understanding
4. **Hands-on practice** - run and analyze real test cases
5. **Reference available** - quick lookup guides for common tasks

## ✨ Highlights

**Unique Features:**
- 5 complete real-world test case walkthroughs
- Python script for automated dependency analysis
- Copy-paste command reference (50+ commands)
- Concrete before/after MLIR examples
- ASCII diagrams for visualization
- FAQ addressing common questions
- Multiple learning paths by time/depth
- Source code cross-references

---

**You're all set!** Pick a learning path from **ACDG_README.md** and start exploring. The tools and materials work together to provide comprehensive coverage from concepts to implementation.
