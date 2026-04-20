# Research Directions on Top of AIR → CSL

The AIR-to-CSL lowering path is working end-to-end: hand-written or lowered AIR goes through the CSL dialect, emits `pe.csl` / `layout.csl` / `run.py`, and runs on the CS-3 simulator. With that foundation in place, this file is a one-liner idea dump — PACT-style paper seeds and research directions spanning frontends, codegen, analysis, applications, systems, and verification. Kept intentionally terse so coverage stays broad.

## Frontends — what else can we lower onto AIR/CSL

- Triton → AIR → CSL; reuse Triton's tile-level abstractions to make GPU programmers instantly productive on WSE.
- Pallas (JAX) → AIR → CSL; one codebase targets both TPU/GPU and Cerebras.
- PyTorch 2.x / ATen → AIR → CSL; end-to-end model compilation for inference on WSE.
- JAX / XLA HLO → AIR → CSL; HLO kernels auto-lowered with shardings preserved.
- MLIR `linalg` (named + generic) → AIR → CSL; the canonical structured-ops frontend.
- Halide schedule language → AIR → CSL; stencil / image kernels with ported schedules.
- TACO (sparse tensor algebra IR) → AIR → CSL; sparse kernels compiled from einsum-like DSL.
- Futhark → AIR → CSL; functional array language targeting WSE.
- Mojo → AIR → CSL; modern systems-Python lowered to spatial dataflow.
- Fortran (via flang/F18) → AIR → CSL; preserve the enormous HPC Fortran codebase.
- OpenMP target offload → AIR → CSL; pragma-annotated C/C++ kernels on WSE.
- SYCL → AIR → CSL; SYCL kernels run on a non-SYCL accelerator via AIR lift.
- OpenACC → AIR → CSL; directive-driven porting of legacy codes.
- DaCe (SDFG) → AIR → CSL; data-centric IR directly mapped to WSE dataflow.
- Domino / SPIRAL → AIR → CSL; signal-processing and transform kernel generation.
- NumPy / CuPy array API → AIR → CSL via a tracing JIT (à la JAX).
- ISL / polyhedral SCoPs → AIR → CSL; polyhedral schedules as WSE placements.
- Stencil-specific DSLs (Exastencils, Stella, Psyclone) → AIR → CSL.
- CuTe / CUTLASS layouts → AIR → CSL; direct layout algebra mapping.
- WebGPU / WGSL compute shaders → AIR → CSL; bring browser compute to WSE.

## Autotuning, autoscheduling, compile-time search

- ML-guided autoscheduling for WSE layouts (RL / XGBoost cost model) — classic Ansor-style research adapted to PE grids.
- Static-vs-runtime DSD parameter autotuning — search comptime stride/offset vs `@set_dsd_*` emission for best latency.
- Placement autotuning across subgrids — pick W×H that minimizes fabric congestion for a given kernel shape.
- Bayesian-optimization-driven sharding — given a memref, pick the shard layout.
- Genetic search over loop nest permutations emitted into CSL.
- Learned cost model for per-PE compute vs memory-bound regime.
- Auto-vectorization width selection (SIMD-16 / -32 / -64) as a tuning knob.
- Profile-guided recompilation loop (run → trace → recompile with hints).
- Multi-objective Pareto search (latency × energy × area-utilization) at compile time.
- Meta-learning across kernels — reuse tuning decisions from past kernels on related shapes.

## Sparse & irregular workloads

- Sparse GEMM / SpMV compilation to WSE with compile-time format selection (COO/CSR/DCSR/block).
- Compressed sensing / sketching kernels on WSE with bespoke layouts.
- Graph neural network (GCN/GAT/GraphSAGE) kernel compilation onto WSE fabric.
- Breadth-first search and other graph-traversal primitives on WSE.
- Sparse attention (Longformer-style banded) compilation to strided mem4d DSDs.
- Mixture-of-Experts routing compiled as dynamic PE-partitioning.
- Hashmap/hashjoin primitives on WSE with load-balanced shard layouts.
- Radix / bitonic sort on WSE — dataflow-first sorting network.
- Sparse Cholesky / LU factorization compilation to WSE.
- Irregular mesh simulation (unstructured FEM) codegen on WSE.
- Top-k selection / approximate nearest neighbor search compiled to WSE.
- Dynamic graphs (streaming updates) on WSE via compile-time rewire passes.

## Performance modeling & analysis

- Analytical roofline model for WSE kernels, parameterized by DSD patterns.
- Fabric-traffic heatmap derivation from AIR IR — visualize congestion before compilation.
- Power / energy model integrated into the AIR-to-CSL pipeline.
- Statically bound per-PE memory-pressure analysis across kernel region.
- Critical-path analysis on the PE dataflow graph; identify schedule slack.
- Static deadlock detection for future color/route scheduling.
- Analytical latency + throughput prediction per kernel from IR alone.
- Network-calculus model for wavelet rate through colors (when fabric lands).
- Polyhedral analysis producing upper-bound execution cycles per subgrid.
- Cache-oblivious compilation for WSE (treat fabric hops as a cache hierarchy).

## Scientific / HPC applications

- Seismic imaging (RTM, FWI) kernels on WSE with layout-optimized stencils.
- Weather / climate kernels (WRF / ICON microphysics) ported via AIR → CSL.
- Computational chemistry (DFT, coupled-cluster tensor contractions) on WSE.
- Molecular dynamics (LAMMPS-style pair kernels) on WSE.
- Lattice QCD on WSE with multi-dim DSD access patterns.
- CFD kernels (PIC, Lagrangian-Eulerian advection) on WSE subgrids.
- Multigrid solvers on WSE with hierarchical PE-grid coarsening.
- Finite-element assembly kernels on WSE.
- Monte Carlo particle transport on WSE with per-PE independent streams.
- Quantum circuit simulators (statevector / tensor-network) on WSE.
- Genomics: sequence alignment, BWT construction, k-mer counting on WSE.
- Cryptographic primitives (SHA-3, lattice-based KEMs) on WSE.
- Digital signal processing pipelines (SDR receivers) on WSE.
- Astrophysical N-body on WSE; explore fabric-as-tree-traversal.
- Protein structure prediction inner kernels (attention-over-residues) on WSE.

## ML systems specifically

- Full transformer inference (GPT-class) compiled from PyTorch to AIR-to-CSL.
- Training pipelines with gradient checkpointing expressed as compiler passes.
- KV-cache layout autotuning for long-context models on WSE.
- Speculative decoding as a compiler-scheduled multi-model mini-pipeline.
- Rotary positional embeddings implemented as strided DSD patterns.
- Quantization-aware lowering (INT8/INT4/NF4) via AIR datatypes → CSL.
- Activation-recomputation policies chosen by the compiler.
- Pipeline parallelism across multiple CS-3 systems via AIR-to-CSL scaling pass.
- Tensor parallelism across PE grid (row/column partitioning of weights).
- Mixture-of-depths / early-exit scheduling.
- Diffusion-model sampler kernels on WSE.
- RAG retrieval + rerank compiled as a unified dataflow graph.
- LLM batching / continuous batching on WSE at compile time.
- Finetuning adapters (LoRA, IA³, DoRA) compiled to per-kernel deltas.
- Reinforcement-learning inner loops (PPO rollouts) on WSE.

## Programming models & abstractions

- A LEGO-like first-class layout algebra contributed upstream to MLIR memref dialect.
- Effect-typed DSD API that prevents bank conflicts at the type level.
- Capability-based memory safety for multi-PE programs — proven at compile time.
- Race-free tasking model above CSL colors/tasks (once fabric lands).
- Linear-types discipline for buffer ownership across PE boundaries.
- APL / J-style functional array DSL as an AIR frontend.
- Partitioned Global Address Space (PGAS) language on WSE (UPC-like).
- Dependent-shape types checked at AIR level (no dyn-shape runtime checks on device).
- Actor model above CSL tasks.
- Lightweight CSP primitives for PE-to-PE channels.

## Verification & correctness

- SMT-based verification of layout transformations (shape, stride preservation).
- Formal bisimulation between AIR program and emitted CSL.
- Static bounds checking on DSD `.extent` / `.offset` / `.stride` combinations.
- Bounded model checking for deadlock-freedom across color routing graphs.
- Differential testing — run the same kernel on GPU and WSE, compare outputs.
- Fuzzing the AIR-to-CSL pipeline (randomized IR → expect emission or clear failure).
- Mechanized semantics of CSL in Coq/Lean for a soundness proof.
- Taint analysis / data-flow verification across PE boundaries.
- Property-based tests generating arbitrary strided layouts and checking emission.
- Symbolic execution of WSE kernels (path enumeration through scf.if branches).

## Debugging & tooling

- Source-level debugger for WSE; compiler-emitted debuginfo maps back to AIR.
- Profiler integration that maps perf counters back to AIR ops and original source.
- Interactive visualization of PE placement + fabric dataflow from AIR IR.
- Time-travel replay for WSE programs.
- AIR-level linter that flags anti-patterns before compilation.
- Compiler-in-the-loop for Jupyter notebooks — JIT kernels straight from a cell.
- Automatic synthesis of host-side `run.py` test drivers from AIR function signatures.
- LSP backend for AIR providing go-to-def / find-references across wafer boundaries.
- `perf annotate` style annotation of emitted CSL with originating IR op.

## Multi-wafer, distributed, systems

- Automatic partitioning across multiple CS-3 wafers at the AIR level.
- Collective-communications library (all-reduce, all-gather) compiled into inter-wafer CSL fabric code.
- Host-device staging autotuner balancing H2D/D2H with compute.
- Fault tolerance — PE-failure-aware compilation (redundant placement).
- Elastic PE allocation so multiple jobs share a wafer.
- Checkpointing strategies chosen by compiler to minimize I/O.
- Co-scheduling CPU/GPU pre/postprocessing with WSE compute stages.
- Transparent migration of hot kernels between wafers during a long run.

## Dataflow & architectural studies

- Empirical comparison GPU SIMT vs WSE dataflow for the same kernels.
- Microbenchmark corpus driven by the emitter (stride vs bank-access latency study).
- Memory-bank conflict analysis across access patterns emitted from subviews.
- Communication locality metrics for a given AIR program.
- Energy-delay Pareto characterization across the v5 test corpus.
- Instruction mix / utilization analysis tool.

## Cross-architecture portability

- Single-source AIR kernel that lowers to GPU, AIE, and WSE — survey paper + tooling.
- Performance-portability study across AMD/NVIDIA/Cerebras via AIR as common ground.
- MLIR bridge passes between LEGO layout IR and AIR memref layouts (mechanized equivalence).
- Retargeting Triton → multiple dataflow backends via AIR as lingua franca.

## Inspirations from the CPU / GPU / FPGA compiler histories

- CUDA → cuBLAS / cuDNN / Thrust → Triton. An analogous stack for WSE: csl-blas / csl-dnn / csl-tensor / WSE-Triton.
- ATLAS / FFTW-style empirical auto-libraries generated per-kernel at install time.
- Polyhedral (Pluto, ISL) tile + skew transformations adapted to PE grids.
- Auto-vectorization research (LLVM SLP, LoopVectorize) re-examined as auto-DSD-ization.
- TVM Ansor + AutoTVM analogues for WSE.
- Halide's compute-at / store-at separation expressed in AIR placement.
- OpenMP worksharing constructs compiled to PE subgrids.
- Legion / Regent partitioning expressed as AIR `place over [lo:hi, lo:hi]`.
- MPI → fabric-colors translation for legacy SPMD codes.
- HLS (Vivado HLS) unroll/pipeline pragmas adapted to AIR annotations.
- `cudaMemcpyAsync` style overlap-compute-and-copy compilation for WSE host staging.
- cuTensor contraction-path DP adapted to WSE multi-dim DSDs.
- WebGPU-like unified-memory programming model on top of WSE.

## "Open problems" papers — survey / position

- Spatial-dataflow compilation: a unified view across FPGAs, AIEs, and WSE.
- "What the WSE is good for, compiler-side": taxonomy paper of which patterns win.
- Retrospective on a decade of tensor-compiler research through a WSE lens.
- Road-map paper: from AIR-to-CSL to a production compiler stack.
- Position: verified compilation is mandatory for reliable wafer-scale computing.
- Reproducibility / artifact-track paper: ship the full AIR → CSL toolchain + benchmark harness.

## Meta / compiler-engineering ideas

- MLIR-dialect-as-a-textbook: AIR + CSL as a teaching vehicle for spatial compilation courses.
- Dataset paper: a benchmark corpus of AIR kernels with CSL ground truths.
- Open-source kernel library (CSLKern) mirroring what cuDNN did for GPUs.
- Property-based IR migration infra: evolve AIR/CSL dialects with automatic test regeneration.
- "Compiler tests as ground truth": using the e2e corpus as a specification.

---

# V2 — Refined research directions

The ideas above are mostly engineering scope: "lower one more frontend", "port one more kernel". Committees don't fund engineering. This section reframes around research questions that the wafer-scale compiler uniquely makes answerable — each idea names the **question**, the **falsifiable hypothesis**, why the **community doesn't know the answer yet**, and a rough **evaluation plan**. Each is intentionally narrow enough to be a single paper.

## 1. SIMD-vs-MIMD granularity autopartitioning

**Question.** On an architecture where 900K PEs can either all run the same program (SIMD) or each run different code (MIMD), what's the compiler policy that picks the right granularity per kernel?

**Hypothesis.** Optimal granularity is statically predictable from two IR-level features: (i) average per-PE working-set reuse, and (ii) control-flow divergence density. A simple classifier beats always-SIMD and always-MIMD by ≥20% mean speedup across a representative kernel corpus.

**Why open.** GPUs are structurally SIMT — no compiler has to decide. CPUs have only MIMD. WSE is the first mainstream machine where the compiler *must* choose, and no existing work characterizes the cost curve.

**Eval.** Build a partitioning pass in AIR with both modes; sweep a 30-kernel corpus (dense/sparse/stencil/graph); measure latency and PE utilization; train a classifier on IR features; compare against always-SIMD and always-MIMD baselines and a hand-tuned Pareto front.

## 2. Cross-architecture cost model grounded in a shared IR

**Question.** Can a single learned cost model, trained on AIR IR, predict kernel runtime on GPU, AIE, and WSE simultaneously — and does performance-portability emerge from a shared feature representation?

**Hypothesis.** Architecture-specific runtime can be factored as a shared "kernel-shape" latent times a target-specific head. Training on a few hundred kernels × 3 targets lets the model predict a fourth target (left out) with <20% relative error.

**Why open.** Every cost-model paper trains per-target. Transfer across fundamentally different architectures (SIMT vs spatial-dataflow vs distributed dataflow) is unexamined. AIR is the rare IR that's literally compiled to all three.

**Eval.** 200-kernel corpus lowered through AIR to GPU/AIE/WSE. Measure actual runtime on each. Train a multi-task model on (IR features, target-descriptor) → runtime. Ablation: which features transfer, which are target-specific? Zero-shot leave-one-target-out evaluation.

## 3. Communication-complexity lower bounds for wafer-scale algorithms

**Question.** For canonical BLAS-3 kernels (GEMM, Cholesky, QR) on a 2-D mesh of N PEs with local-only fabric, what is the provable lower bound on fabric traffic — and how close does a compiler-generated placement get?

**Hypothesis.** Classical I/O lower bounds (Hong–Kung, Kwasniewski) give bandwidth bounds assuming a flat memory; the mesh-topology WSE variant admits a stricter bound. Compiler placements from an ILP-driven pass achieve within a constant factor of this bound.

**Why open.** Lower bounds for spatial architectures with local-only communication are published only for synchronous systolic arrays — not for the asynchronous, per-PE-programmable model CS-3 actually exposes. This is the theoretical gap.

**Eval.** Theorem + matching upper bound (constructive) via a compiler pass. Measure real fabric traffic via performance counters. Compare against hand-coded reference kernels.

## 4. Is the strided-layout algebra expressively complete for WSE-reachable access patterns?

**Question.** The `(shape, stride, offset)` algebra used by CSL mem1d/mem4d DSDs and upstream MLIR can express many access patterns — but *which* patterns? Morton-order? Hilbert-curve? Bit-reversal permutations?

**Hypothesis.** There exists a natural class of access patterns (parameterizable affine + "bit-permutation" pattern family) that WSE hardware can execute in one DSD instruction, but that the compiler cannot *synthesize* within the strided-layout algebra — forcing explicit loops and 10×+ slowdowns.

**Why open.** CuTe / LEGO / MLIR strided layouts are widely used but nobody has formally characterized their expressiveness gap against hardware capabilities. A completeness theorem would guide the next generation of layout IRs.

**Eval.** (i) Proof: which bit-permutation patterns admit a single-DSD representation? (ii) Empirical: implement patterns outside the class via loops vs inside via DSDs; measure slowdown. (iii) Propose a conservative extension that closes the gap.

## 5. Deterministic-by-construction numerics on spatial dataflow

**Question.** Floating-point non-associativity makes GPU kernels numerically irreproducible across scheduler decisions. Can a spatial compiler *prove* deterministic numerics by fixing the reduction tree at compile time, at zero runtime cost?

**Hypothesis.** For any reduction expressible in AIR, the compiler can emit a placement whose fabric ordering is deterministic under hardware non-preemption guarantees. The compiler-certified version suffers ≤5% slowdown vs the fastest non-deterministic version across an ML-training benchmark suite.

**Why open.** Reproducibility is a first-order concern for scientific computing and ML validation; existing solutions (atomic-free reductions, deterministic cuDNN modes) are per-kernel patches. A compiler-wide guarantee doesn't exist for GPUs (preemption + warp-scheduling aren't controllable). WSE's static scheduling *does* make it possible.

**Eval.** Instrument the AIR-to-CSL pipeline with a determinism-certification pass. Validate bit-identical outputs across 1000 runs of a 20-workload suite. Measure the determinism tax.

## 6. Algorithm-architecture co-design: which algorithms change class?

**Question.** Which numerical algorithms move from "impractical at scale" to "competitive" when the memory bandwidth bottleneck is replaced by on-chip fabric? Conversely, which "GPU-winners" become losers?

**Hypothesis.** Algorithms with O(N^{1+ε}) communication — sparse iterative solvers, Barnes-Hut, distributed Monte Carlo — cross a threshold into competitiveness on WSE, while dense kernels with strong arithmetic intensity (GEMM, FFT) lose their relative advantage. A predictive model built on (arithmetic intensity, communication volume) can classify algorithms ex ante.

**Why open.** Arithmetic intensity as a prediction tool was calibrated on GPUs (Williams et al. 2009). WSE's communication profile invalidates the ordinal ranking — this is uncharted territory.

**Eval.** Take a suite of 10 algorithms with known GPU baselines, compile to WSE, measure. Report the performance-rank reshuffle. Build a predictive model and evaluate on a held-out algorithm.

## 7. Compiler-scheduled fault tolerance as a first-class compilation target

**Question.** With ~900K PEs, single-PE failures are inevitable at scale. Can a spatial compiler bake in k-redundancy at compile time — placing replicas such that any single failure is masked with zero detection latency — and what's the area / energy cost?

**Hypothesis.** A constrained placement pass can enforce k=2 redundancy for any AIR kernel at ≤2.5× area (vs naive duplication's 2×) via co-placement optimizations, and <1% runtime overhead.

**Why open.** Existing accelerator fault tolerance is runtime (ECC, checksum) or static (triple-modular redundancy in space). Neither considers the compiler's freedom over placement. No published wafer-scale compiler paper addresses this.

**Eval.** Extend the placement pass with a redundancy constraint. Inject faults via simulator; measure detection latency and masked-result correctness. Characterize the cost curve as a function of k.

## 8. Memory hierarchy as a lattice, not a tree

**Question.** Classical tiling theory assumes a strict cache hierarchy (L1 ⊂ L2 ⊂ L3 ⊂ DRAM). WSE's memory model is each-PE-to-neighbor with bounded-hop fabric — a lattice, not a tree. Does this break polyhedral tile-selection heuristics in predictable ways, and what replaces them?

**Hypothesis.** Polyhedral cost models (Pluto-style) systematically over-tile on WSE because they assume a reuse-within-tile model that fabric-neighbor locality violates. A reformulation in terms of graph-geodesic distance in the PE mesh recovers prediction accuracy within 10% of measured.

**Why open.** Polyhedral compilation has a 30-year literature, all tree-hierarchical. Lattice-hierarchy reformulation is not just a knob change; it's a structural one. WSE is the first commercial target that exposes this gap.

**Eval.** Port 15 stencil/affine kernels through Pluto → AIR → CSL, then again through a proposed lattice-aware scheduler; compare predicted vs measured runtime.

## 9. The verified-lowering thesis

**Question.** Can we mechanize the correctness of AIR → CSL lowering in Lean/Rocq such that the emitted CSL provably preserves a source-level denotational semantics — and is the resulting compiler pipeline usable in production for safety-critical HPC (e.g. climate, reactor sim)?

**Hypothesis.** A compositional operational semantics for a strict AIR subset plus CSL memory DSDs admits a mechanized soundness proof with proof-engineering effort comparable to CompCert's 6 years × 2 FTEs, scaling sublinearly in dialect additions.

**Why open.** CompCert verified C-to-assembly. No verified compiler exists for spatial / dataflow architectures. The WSE is a strong target because its execution model is more deterministic than a multicore CPU — actually makes proofs tractable.

**Eval.** Formalize CSL memory semantics (mem1d_dsd only) in Lean 4. Prove semantic preservation for the scientific-corpus kernels. Quantify proof effort vs pipeline lines of code.

## 10. Compiler-generated kernel libraries vs hand-tuned: is the gap closing?

**Question.** For a given per-PE compute budget, does a compiler-search-generated kernel match or beat the hand-tuned SDK reference on a well-chosen benchmark suite — and if not, what's the residual human edge?

**Hypothesis.** After an overnight autotuning budget per kernel, compiler-generated kernels match hand-tuned on ≥70% of a 20-kernel suite; the remaining 30% share a common structural property (e.g., data-dependent control flow) that future compiler features can address.

**Why open.** This question has been asked for GPUs for a decade (Halide, TVM, Triton). For WSE, the hand-written reference corpus exists but no automated search has been measured against it. The *comparison itself* is the contribution.

**Eval.** Build the search harness (random + evolutionary + learned cost model). Run on SDK reference kernels. Characterize the 30% gap with structural analysis of the kernels.

---

## Shortlist to defend as a single PACT submission

If forced to pick **one**, I'd back **#3 (communication-complexity lower bounds)** or **#6 (algorithm-architecture co-design)**. Both have:

- A theoretical or methodological contribution that lives past the WSE-specific artifact.
- Clear quantitative evaluation.
- Hardware access is a real differentiator (no one else can reproduce easily).
- Teach the community something about *architectures*, not just about *our compiler*.

#1 (SIMD-vs-MIMD) is the best "pure PACT" paper — the question is the sort of thing the community would love to see a data-driven answer to, and the architectural opportunity is genuinely new.

#9 (verified lowering) is the best "decade-horizon" paper — smaller audience now, but higher citation trajectory if Cerebras enters safety-critical domains.

Everything in V1 above is available as *supporting infrastructure* for whichever of these is pursued.

---

# V3 — same questions, different venues

Each conference rewards different framings. Same underlying wafer-scale compiler opens different research angles depending on where you pitch it. Below: per-venue idea clusters. Ideas with ✦ are the strongest fit for that venue.

## SC (SuperComputing) — scale, applications, energy

SC cares about: running a real scientific code at scale, establishing a benchmark position, energy-efficiency at exascale equivalent, or doing something hard and measurable on hardware that matters.

- ✦ **HPCG + HPL on CS-3 via AIR compiler.** Establish the WSE's position in canonical HPC benchmarks. The **measurement methodology** for a machine that breaks the standard distributed-memory model (no DRAM hierarchy per node) is itself the contribution.
- ✦ **Climate/weather grand-challenge port.** Take a non-trivial dynamical core (e.g. ICON tracer transport, WRF microphysics) end-to-end through AIR→CSL. Strong-scale on single + multi-wafer; report J/simulated-year vs a GPU cluster. SC’s bread and butter.
- **Exascale energy proportionality.** Instrument the pipeline with per-kernel energy measurements; derive an energy-model paper: "how much is the compiler’s choice worth in joules?"
- **Mixed-precision iterative refinement at wafer scale.** Couple FP64 residual correction to FP32 work kernels; show the numerical science the deterministic-numerics pipeline enables on WSE but not GPU.
- **Sparse iterative solvers (PCG, GMRES) co-designed with PE placement.** SC Gordon-Bell material when the port wins against a GPU cluster on a production problem.
- **Wafer-to-wafer collectives as first-class citizens.** SC loves communication libraries. Derive an all-reduce / all-gather primitive, measure scaling across N wafers.
- **Checkpoint-restart / C/R economics on WSE.** Large problems demand resilience; characterize C/R overhead vs fault rate, propose compiler-scheduled C/R.
- **Reproducible HPC.** The deterministic-numerics idea (V2 #5) reframed: "SC community, here's how you get bit-identical output across runs — impossible on GPUs."
- **Exascale I/O pipeline.** How do you feed 900K PEs? Compiler-scheduled stream-of-tiles from disk; hide I/O behind compute.

## ASPLOS — hardware/software co-design, novel abstractions, systems

ASPLOS wants ideas that span layers: a new abstraction, a proposed hardware change motivated by software, OS/runtime integration, or an architectural observation made compiler-visible.

- ✦ **Compiler-proposed ISA extensions for WSE.** From AIR-level analysis, identify a missing fabric primitive (e.g. bounded-broadcast, dynamic-length DSDs, async-rendezvous) and show via simulation / RTL that adding it would yield N% speedup on an end-to-end workload.
- ✦ **Elastic-PE-allocation runtime.** Several jobs share a wafer; the compiler emits relocatable, rankable placements; an OS scheduler picks tile assignments dynamically. A new resource-management model (vs GPU’s kernel-at-a-time).
- **Unified-memory programming model across host DRAM + wafer fabric.** Language-level shared array; compiler decides staging. Bridge the two memory systems abstractly.
- **Compiler-generated DUE (detected-uncorrectable error) recovery.** Observations on the fault model made actionable via compiler-placed redundancy + OS-level recovery coordination.
- **Fine-grained memory capabilities across PEs.** Security story: compiler-enforced isolation within a wafer shared by multiple tenants.
- **Speculative execution as a compiler-placed primitive.** Execute both branches at different PEs, squash losers — pushes speculation from hardware to software.
- **OS support for compiler-declared real-time kernels on WSE.** Co-design story: compiler annotates latency contracts; OS guarantees placement + isolation.
- **Dynamic re-placement on PE-failure.** Co-design between the compiler's placement IR and a runtime that migrates logical PEs on fault.
- **Transactional fabric writes.** Propose a hardware primitive that ASPLOS cares about (transactions have been an ASPLOS topic for 20 years), measure via compiler-simulated workloads.

## CGO — codegen techniques, compiler algorithms, IR design

CGO wants a novel compiler technique with a measured benefit, an IR contribution, or a classic opt applied to a new target.

- ✦ **A DSD allocator.** DSDs are a bounded per-PE resource (only N can be live). Novel register-allocation-style algorithm treating DSDs as a new kind of resource. Compare graph-coloring, linear-scan, PBQP on this problem.
- ✦ **Compile-time cost-model-driven layout selection.** Search stride/offset/placement triples; evaluate via a lightweight cost model. CGO canonical format: model accuracy plot + speedup plot on a kernel suite.
- **Peephole catalog for emitted CSL.** Empirical paper: instrument the emitter, inventory 30+ peephole rewrites, quantify each’s contribution. This is *exactly* a CGO paper.
- **Static rematerialization policy for DSDs.** When to rebuild a DSD vs keep it; classical remat problem on a new resource. Analytic solution + measured benefit.
- **Instruction-scheduling for 40k-op per-PE budget.** List scheduling vs SWP vs trace scheduling on CSL kernels; which technique wins and why?
- **Compiler testing via differential execution.** AIR → CSL vs AIR → GPU; differential testing as a CGO paper.
- **Incremental AIR→CSL compilation.** Small edit → fast rebuild; critical for autotuning throughput. Cache-analysis + IR-diff-based rebuild.
- **Provably-optimal tile selector.** Polyhedral / ILP formulation of WSE tile selection with an optimality guarantee; compare to heuristics.
- **Dead-code elimination across PE boundaries.** Novel DCE that looks across spatial extent — only possible with AIR’s placement-explicit IR.
- **Automatic partial evaluation of comptime CSL parameters.** A CGO-sized contribution: how much runtime overhead does the compiler save by specializing across comptime params?

## IPDPS — parallel/distributed algorithms, theory, communication

IPDPS has strong theoretical and algorithmic tracks; proof-supported results, new algorithms for a new topology, or rigorous communication analyses all fit.

- ✦ **Lower + matching upper bounds for canonical parallel primitives on a bounded-degree 2-D mesh with PE-local memory.** Prefix-sum, sort, transpose, reduce. Theorem + constructive matching compiler pass + empirical validation. Exact IPDPS format.
- ✦ **Load balancing for irregular workloads on WSE.** Graph algorithms (BFS, PageRank, SSSP) demand dynamic balancing; propose a new algorithm exploiting WSE’s synchronous fabric and prove work-optimality.
- **All-to-all communication algorithms on bounded-degree wafer fabric.** Known GPU algorithms don’t map; derive new ones with provable bandwidth optimality for this topology.
- **Decentralized consensus primitives on WSE fabric.** Byzantine fault tolerance + agreement; classical distributed-systems topic on a new substrate.
- **Parallel PDE solvers with provable convergence on finite-PE meshes.** Numerical analysis paper; bound the convergence-rate penalty from fixed PE allocation.
- **Scheduling theory for WSE jobs.** Multi-tenant scheduling on a shared wafer; online vs offline competitive ratios.
- **Cache-oblivious algorithms retargeted to lattice memory hierarchies.** Generalize the cache-oblivious framework (Frigo–Leiserson) to wafer topology.
- **Energy-aware scheduling theory at 900K-way parallelism.** Minimize J subject to latency deadline; derive the pareto-optimal schedules.

## Bonus clusters (stretch venues)

### HPCA / MICRO — microarchitecture

- **Workload characterization paper.** Instrument emitted CSL; report IPC, fabric utilization, memory-bank contention across the scientific corpus. Data-driven architecture insights.
- **"What would WSE-next have?"** Compiler-observations-driven proposals for the next generation fabric (width, hops, queue depth).

### MLSys — ML systems specifically

- ✦ **Reproducible training for foundation models on WSE.** Bit-identical training runs across fabric reconfigurations — uniquely possible, genuinely demanded by the ML-eval community.
- **Shape-generic LLM inference compilation.** Autoscheduler for dynamic-shape kernels on WSE; MLSys likes shape-genericity.
- **Long-context attention placements.** Attention kernels that win on WSE because the KV cache fits on-chip distributed across PEs.
- **Distributed training without a global clock.** Explore what WSE's synchrony enables for step-size adaptation and curvature estimation.

### PPoPP — parallel programming models

- **A new programming model for wafer-scale dataflow.** Task-parallel + structured synchronization; derive and evaluate against AIR baseline.
- **Debugging at 900K-way parallelism.** A new methodology; compiler-assisted observation of distributed state.

### OOPSLA / POPL — programming languages + semantics

- **Mechanized semantics of CSL mem DSDs.** Small-step operational semantics formalized; a prerequisite for everything in V2 #9 that also stands alone.
- **A type system for safe inter-PE communication.** Session types / linear types for future color-channel CSL.

### DAC / ICCAD — design automation / CAD

- **Compiler-driven wafer-layout synthesis.** Given a workload graph, emit both the kernel placement and (simulated) fabric configuration; full-stack design automation.

## Cross-venue strategy

One underlying result can often be retold for two conferences with different framing:

| Underlying work | Venue A | Venue B |
|---|---|---|
| Deterministic numerics (V2 #5) | SC (reproducible HPC) | MLSys (reproducible training) |
| Comm-complexity lower bounds (V2 #3) | IPDPS (theorem) | SC (practical library) |
| SIMD-vs-MIMD partitioning (V2 #1) | PACT (general) | CGO (compiler technique) |
| Algorithm reshuffle (V2 #6) | PACT (position) | SC (Gordon Bell application) |
| Verified lowering (V2 #9) | OOPSLA (semantics) | CGO (compiler correctness) |
| ISA extensions (V3 ASPLOS #1) | ASPLOS (co-design) | HPCA (architecture) |
| DSD allocator (V3 CGO #1) | CGO (algorithm) | LCTES (embedded code-gen) |

The rule of thumb: SC wants *real workloads + real scale*, ASPLOS wants *layer-spanning novelty*, CGO wants *a compiler algorithm*, IPDPS wants *proofs*, PACT wants *a question*, MLSys wants *ML impact*, PPoPP wants *programming models*. Each of these can be satisfied by subsetting what we already have.

