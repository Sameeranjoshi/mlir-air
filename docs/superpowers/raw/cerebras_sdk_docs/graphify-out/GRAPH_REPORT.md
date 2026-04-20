# Graph Report - .  (2026-04-17)

## Corpus Check
- 184 files · ~98,320 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 635 nodes · 760 edges · 42 communities detected
- Extraction: 89% EXTRACTED · 11% INFERRED · 0% AMBIGUOUS · INFERRED: 81 edges (avg confidence: 0.81)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- [[_COMMUNITY_Tutorial & Example Concepts|Tutorial & Example Concepts]]
- [[_COMMUNITY_Appliance & Runtime Infrastructure|Appliance & Runtime Infrastructure]]
- [[_COMMUNITY_Compiler Flags & Builtins|Compiler Flags & Builtins]]
- [[_COMMUNITY_Modules, Imports & Storage|Modules, Imports & Storage]]
- [[_COMMUNITY_GEMV Tutorials (Memcpy & Params)|GEMV Tutorials (Memcpy & Params)]]
- [[_COMMUNITY_SdkLayout API & Fabric DSDs|SdkLayout API & Fabric DSDs]]
- [[_COMMUNITY_Debugger (CSDB & GUI)|Debugger (CSDB & GUI)]]
- [[_COMMUNITY_Linear Algebra Benchmarks|Linear Algebra Benchmarks]]
- [[_COMMUNITY_Tasks, Colors & Builtins|Tasks, Colors & Builtins]]
- [[_COMMUNITY_DSDs & Memory Operations|DSDs & Memory Operations]]
- [[_COMMUNITY_Microthreads & Task IDs (WSE-3)|Microthreads & Task IDs (WSE-3)]]
- [[_COMMUNITY_Pipelines, DSRs & Mandelbrot|Pipelines, DSRs & Mandelbrot]]
- [[_COMMUNITY_Sparse Solvers (SpMV, PCG, Power)|Sparse Solvers (SpMV, PCG, Power)]]
- [[_COMMUNITY_CSL Introduction & Basic Syntax|CSL Introduction & Basic Syntax]]
- [[_COMMUNITY_Dense Linear Algebra Collectives|Dense Linear Algebra Collectives]]
- [[_COMMUNITY_Switches Tutorials (06, 07)|Switches Tutorials (06, 07)]]
- [[_COMMUNITY_Debug & Simprint Tutorials|Debug & Simprint Tutorials]]
- [[_COMMUNITY_GEMV Checkerboard & Collectives|GEMV Checkerboard & Collectives]]
- [[_COMMUNITY_Sentinels & Control Wavelets|Sentinels & Control Wavelets]]
- [[_COMMUNITY_FFT Benchmarks|FFT Benchmarks]]
- [[_COMMUNITY_Histogram on Torus|Histogram on Torus]]
- [[_COMMUNITY_Debug & Simprint Libraries|Debug & Simprint Libraries]]
- [[_COMMUNITY_Sparse Tensors Tutorial|Sparse Tensors Tutorial]]
- [[_COMMUNITY_Sentinels Tutorial|Sentinels Tutorial]]
- [[_COMMUNITY_Filters Tutorial|Filters Tutorial]]
- [[_COMMUNITY_Color Swap Tutorial|Color Swap Tutorial]]
- [[_COMMUNITY_WSE-3 Microthreads Tutorial|WSE-3 Microthreads Tutorial]]
- [[_COMMUNITY_FIFOs Tutorial|FIFOs Tutorial]]
- [[_COMMUNITY_Map Builtin Tutorial|Map Builtin Tutorial]]
- [[_COMMUNITY_Residual Benchmark|Residual Benchmark]]
- [[_COMMUNITY_RowCol Broadcast Benchmark|Row/Col Broadcast Benchmark]]
- [[_COMMUNITY_Wavelet Payload Filtering|Wavelet Payload Filtering]]
- [[_COMMUNITY_Game of Life Benchmark|Game of Life Benchmark]]
- [[_COMMUNITY_Wide Multiplication Benchmark|Wide Multiplication Benchmark]]
- [[_COMMUNITY_Collectives Library & MPI|Collectives Library & MPI]]
- [[_COMMUNITY_@map Builtin|@map Builtin]]
- [[_COMMUNITY_Color Swap Routing|Color Swap Routing]]
- [[_COMMUNITY_Microthread Queue Decoupling|Microthread Queue Decoupling]]
- [[_COMMUNITY_Bandwidth Test|Bandwidth Test]]
- [[_COMMUNITY_FFT 3D|FFT 3D]]
- [[_COMMUNITY_Row-Col Broadcast|Row-Col Broadcast]]
- [[_COMMUNITY_GEMV-06 Exercises|GEMV-06 Exercises]]

## God Nodes (most connected - your core abstractions)
1. `CSL Code Examples Index` - 22 edges
2. `SDK Release Notes (Cumulative)` - 15 edges
3. `GEMV-06 Device Code (routes)` - 14 edges
4. `GEMV 2 Device Code` - 13 edges
5. `SDK GUI` - 13 edges
6. `CSL Language Guide Index` - 12 edges
7. `CSL Type System` - 12 edges
8. `GEMV 1 Device Code` - 12 edges
9. `Cerebras SDK Documentation README/TOC` - 11 edges
10. `Running SDK on a Wafer-Scale Cluster (Appliance Mode)` - 11 edges

## Surprising Connections (you probably didn't know these)
- `CSL layout block` --semantically_similar_to--> `SdkLayout class`  [INFERRED] [semantically similar]
  computing-with-cerebras.md → api-docs/sdklayout-api.md
- `SdkCompiler Python class` --semantically_similar_to--> `cslc CSL compiler helper script`  [INFERRED] [semantically similar]
  appliance-mode.md → installation-guide.md
- `SdkLauncher Python class` --semantically_similar_to--> `cs_python helper script`  [INFERRED] [semantically similar]
  appliance-mode.md → installation-guide.md
- `SdkRuntime appliance bindings (deprecated)` --semantically_similar_to--> `SdkRuntime host runtime`  [INFERRED] [semantically similar]
  appliance-mode.md → tensor-streaming.md
- `fabin_dsd / fabout_dsd` --shares_data_with--> `Wavelet (32-bit fabric message)`  [INFERRED]
  csl/Language/DSDs.md → computing-with-cerebras.md

## Hyperedges (group relationships)
- **CSL task types trio** — data_task, local_task, control_task [EXTRACTED 0.95]
- **memcpy resource reservation set** — memcpy_infrastructure, color, input_queue, local_task, control_task [EXTRACTED 0.90]
- **Appliance compile-launch-run workflow** — sdk_compiler_class, sdk_launcher_class, sdk_runtime_appliance, wafer_scale_cluster [EXTRACTED 0.90]
- **Three Task Types: Data/Local/Control** — task_ids_data_task, task_ids_local_task, task_ids_control_task [EXTRACTED 1.00]
- **CSL Struct Type Family** — types_anon_struct, types_named_struct, types_comptime_struct [EXTRACTED 1.00]
- **Iterative Linear Solver Benchmarks** — bench_cg, bench_pcg, bench_bicgstab, bench_power_method [INFERRED 0.85]
- **GEMV tutorial progression (00-09)** — gemv_00_basic_syntax, gemv_01_complete_program, gemv_02_memory_dsds, gemv_03_memcpy, gemv_04_params, gemv_05_multiple_pes, gemv_06_routes_1, gemv_07_routes_2, gemv_08_routes_3, gemv_09_streaming [INFERRED 0.90]
- **Pipeline tutorial series** — pipeline_01_basic, pipeline_02_fifo, pipeline_03_multiple [INFERRED 0.90]
- **SdkLayout tutorial series** — sdklayout_01_introduction, sdklayout_02_routing, sdklayout_03_ports_connections, sdklayout_04_h2d_d2h, sdklayout_05_gemv [INFERRED 0.90]
- **Iterative linear solvers using 7-pt stencil SpMV** — bench_bicgstab, bench_conjugate_gradient, bench_7pt_stencil_spmv, concept_stencil_3d_7pts_pe, concept_allreduce_pe [EXTRACTED 0.95]
- **Timing synchronization pattern (tic/toc/sync)** — bench_7pt_stencil_spmv, bench_bandwidth_test, bench_bicgstab, bench_conjugate_gradient, concept_host_callable_sync, concept_tsc_counter_sync [EXTRACTED 0.95]
- **FFT family of benchmarks** — bench_fft_1d_2d, bench_fft_3d, concept_cooley_tukey_dit [EXTRACTED 0.90]
- **GEMV tutorial progression (0-9)** — gemv_00_basic_syntax, gemv_01_complete_program, gemv_02_memory_dsds, gemv_03_memcpy, gemv_04_params, gemv_05_multiple_pes, gemv_06_routes_1, gemv_07_routes_2, gemv_08_routes_3, gemv_09_streaming [EXTRACTED 1.00]
- **Stencil-based iterative solvers sharing infrastructure** — power_method, pcg, allreduce_pe, stencil_3d_7pts_pe [EXTRACTED 1.00]
- **Parallel GEMV strategies on PE rectangle** — gemv_checkerboard_pattern, gemv_collectives_2d, residual, gemv_08_routes_3 [INFERRED 0.80]
- **SdkLayout tutorial progression** — sdklayout_01_introduction, sdklayout_02_routing, sdklayout_03_ports_connections, sdklayout_04_h2d_d2h, sdklayout_05_gemv [EXTRACTED 0.95]
- **Pipeline FIFO resource trade-off evolution** — pipeline_02_fifo, pipeline_02_fifo_resource_disadvantage, pipeline_03_multiple, pipeline_03_halo_rationale [EXTRACTED 0.90]
- **Control-wavelet family of features** — topic_05_control_wavelet, topic_06_control_library, topic_07_control_task, topic_05_sentinels, topic_06_switches, topic_07_switches_entrypt [INFERRED 0.80]
- **Memcpy-based program flow (host launches kernel, device computes, host copies result)** — concept_sdkruntime, concept_memcpy_lib, concept_unblock_cmd_stream, concept_memcpy_d2h, concept_pe_program_csl [EXTRACTED 0.90]
- **DSD-based GEMV using memory DSDs and builtins** — concept_memory_dsd, concept_mem1d_dsd, concept_fmacs, concept_fadds, concept_increment_dsd_offset [EXTRACTED 0.95]
- **Layout + PE program export/import pattern** — concept_layout_csl, concept_pe_program_csl, concept_set_rectangle, concept_set_tile_code, concept_export_name, concept_export_symbol [EXTRACTED 0.90]
- **GEMV-03 three-phase flow (H2D, kernel, D2H)** — memcpy_h2d, memcpy_d2h, three_phase_program, sdk_runtime [EXTRACTED 0.90]
- **Multi-PE layout configuration pattern** — set_rectangle, set_tile_code, memcpy_multi_get_params, program_rectangle, width_param [EXTRACTED 0.85]
- **Async fabric DSD transfer pattern with task activation** — fabout_dsd, fabin_dsd, fmovs, fadds, exit_task_id, bind_local_task [EXTRACTED 0.90]
- **CSDB Debug Workflow (context, target, rectangle, memory)** — tool_csdb, csdb_context_command, csdb_target_command, csdb_rectangle_command, csdb_memory_command [EXTRACTED 0.95]
- **SIMFABRIC_DEBUG Log Modes** — env_SIMFABRIC_DEBUG, log_landing, log_inst_trace, log_router [EXTRACTED 0.95]
- **SDK GUI Timeline and Trace Panels** — gui_timeline_wavelet, gui_timeline_instruction, gui_timeline_combined, gui_wavelet_traces, gui_instruction_traces [EXTRACTED 0.90]

## Communities

### Community 0 - "Tutorial & Example Concepts"
Cohesion: 0.03
Nodes (79): CSL Code Examples Index, Color as route identifier, Arrays and Pointers in CSL, FIFO (First-In-First-Out buffer), GEMV (General Matrix-Vector Multiply), CSL Libraries (imports), Memcpy (host-device transfer), pe_program.csl (per-PE program) (+71 more)

### Community 1 - "Appliance & Runtime Infrastructure"
Cohesion: 0.05
Nodes (61): Advanced Hardware Features, anytype generic mechanism, SDK Appliance API Reference, Appliance Logging (cerebras.appliance.logger), Running SDK on a Wafer-Scale Cluster (Appliance Mode), CE Inject Mode (WSE-2 only), CodeRegion, Color Swapping (+53 more)

### Community 2 - "Compiler Flags & Builtins"
Cohesion: 0.05
Nodes (53): --arch flag (wse2/wse3), --channels flag, SdkRuntime cmaddr argument, comptime block, @constants builtin, CS-2 System (WSE-2), CS-3 System (WSE-3), cs_python (+45 more)

### Community 3 - "Modules, Imports & Storage"
Cohesion: 0.06
Nodes (42): CSL Modules, @import_module builtin, imported_module Type, param_binding, Standard Library Import (angle brackets), Storage Classes, Python ELFLoader class, export Storage Class (+34 more)

### Community 4 - "GEMV Tutorials (Memcpy & Params)"
Cohesion: 0.07
Nodes (40): Compile-time parameters, cslc --params flag, GEMV-03 Exercises, GEMV-03 Host Code (memcpy_h2d), GEMV Tutorial 3: Memcpy, GEMV-03 Learning Objectives, GEMV-03 Overview, GEMV-03 Compile and Run (+32 more)

### Community 5 - "SdkLayout API & Fabric DSDs"
Cohesion: 0.08
Nodes (38): CSELFRunner (deprecated, removed), SdkLayout Program Layout API (beta), SdkRuntime Host Runtime, WSE-3 Architecture, Asynchronous Builtin Operations on Fabric DSDs, cb16 / bfloat16 FP Types, fabin_dsd (Fabric Input DSD), fabout_dsd (Fabric Output DSD) (+30 more)

### Community 6 - "Debugger (CSDB & GUI)"
Cohesion: 0.07
Nodes (32): Coredump Inspection (corefile.cs1), memcpy Reserved Colors and Queues, CSDB context command, CSDB memory command, CSDB rectangle command, CSDB settings command, CSDB target command, CSDB trace command (+24 more)

### Community 7 - "Linear Algebra Benchmarks"
Cohesion: 0.09
Nodes (29): Benchmark: 25-pt Stencil, Benchmark: 25-Point Stencil, Benchmark: 7-pt Stencil SpMV, Benchmark: 3D 7-Point Stencil SpMV, Benchmark: Bandwidth Test, Benchmark: BiCGSTAB, Benchmark: Conjugate Gradient, Benchmark: Conjugate Gradient (+21 more)

### Community 8 - "Tasks, Colors & Builtins"
Cohesion: 0.08
Nodes (28): @activate builtin, @bind_control_task builtin, @bind_data_task builtin, Block/Unblock task semantics, CSL Builtins, CSL Builtins for WSE-3, Color (virtual communication channel), <complex> library (+20 more)

### Community 9 - "DSDs & Memory Operations"
Cohesion: 0.11
Nodes (24): Fabric DSDs must be async, @bind_local_task builtin, Column-major storage of A, Memory DSD, @fmacs and @fadds DSD operations, exit_task_id, fabin_dsd, fabout_dsd (+16 more)

### Community 10 - "Microthreads & Task IDs (WSE-3)"
Cohesion: 0.12
Nodes (18): Asynchronous DSD Operation, @block / @unblock builtins, @get_ut_id builtin, @load_to_dsr builtin, Microthread Operand Priority (dest > src1 > src2), ut_id Type, Microthread IDs (WSE-3), Activatable Identifiers (+10 more)

### Community 11 - "Pipelines, DSRs & Mandelbrot"
Cohesion: 0.12
Nodes (17): CSL Appendix, CSL Pipeline pattern, Data Structure Descriptors (DSDs), DSR types (dsr_dest/src0/src1/fifo), Data Structure Registers (DSRs), fabin_dsd / fabout_dsd, Mandelbrot, Known problems: load balancing, iters stored as f32 (+9 more)

### Community 12 - "Sparse Solvers (SpMV, PCG, Power)"
Cohesion: 0.13
Nodes (17): allreduce2R1E/pe.csl (2 colors, 1 entrypoint), Rationale: spmv kernel has only three unused colors, allreduce/pe.csl (rectangle reduction for sync), hypersparse_spmv/pe.csl (2D-partitioned spmv), Preconditioned Conjugate Gradient, Jacobi preconditioner, kernel.csl (spmv, dot, updates), kernel_pcg.csl (f_pcg state machine on WSE) (+9 more)

### Community 13 - "CSL Introduction & Basic Syntax"
Cohesion: 0.12
Nodes (17): CSL Builtins, CSL Examples GitHub Repository, CSL Basic Types, For/While Loops, Processing Element (PE), @range builtin, var and const keywords, Wafer Scale Engine (WSE) (+9 more)

### Community 14 - "Dense Linear Algebra Collectives"
Cohesion: 0.12
Nodes (16): Benchmark: Cholesky, Benchmark: GEMM Collectives, Benchmark: GEMV Checkerboard, Benchmark: GEMV Collectives, Benchmark: Single-Tile Matvec, Cholesky decomposition (right-looking), collectives_2d library, Fabric switches / control wavelets (+8 more)

### Community 15 - "Switches Tutorials (06, 07)"
Cohesion: 0.17
Nodes (12): topic-06 empty.csl, topic-06 layout.csl, topic-06 recv.csl, topic-06 run.py, topic-06 send.csl, Tutorial 06: Switches, topic-07 empty.csl, topic-07 layout.csl (+4 more)

### Community 16 - "Debug & Simprint Tutorials"
Cohesion: 0.2
Nodes (10): Tutorial 12: Debug Library, topic-12 layout.csl, topic-12 receiver.csl, topic-12 run.py, topic-12 sender.csl, topic-13 layout.csl, topic-13 receiver.csl, topic-13 run.py (+2 more)

### Community 17 - "GEMV Checkerboard & Collectives"
Cohesion: 0.2
Nodes (10): Rationale: cannot safely send/receive multiple wavelets on same color with fixed routing, Matrix A distribution across 4x4 PE kernel, FP16 computation (checkerboard GEMV), GEMV with Checkerboard Pattern, Rationale: intended as non-trivial introductory example, not optimized, GEMV with Collective Communications, FP32 computation (collectives GEMV), Rationale: non-trivial introductory example of collectives library (+2 more)

### Community 18 - "Sentinels & Control Wavelets"
Cohesion: 0.25
Nodes (8): Control wavelet, Sentinel control task ID signals end of input tensor, Topic 5: Sentinels, <control> library encode_single_payload, Topic 6: Switches, Switches save colors and allow limited runtime route control, Control task activated by control wavelet task ID, Topic 7: Switches and Control Entrypoints

### Community 19 - "FFT Benchmarks"
Cohesion: 0.33
Nodes (7): Benchmark: 1D/2D FFT, Benchmark: 3D FFT, Bit-reversed indexing / butterfly, Cooley-Tukey DIT radix-2, Pencil decomposition, Twiddle factors, Rationale: data preloaded, no halo needed

### Community 20 - "Histogram on Torus"
Cohesion: 0.29
Nodes (7): Bucket Distribution (row-major, NUM_BUCKETS per PE), Rationale: post-route encoding eases X ID extraction without masking, Rationale: no PE knows global completion, tally column aggregates, Two-hop N/S then E/W routing, Tally kernel (termination column), HISTOGRAM on Torus, Wavelet encoding (Y/X/bucket bits)

### Community 21 - "Debug & Simprint Libraries"
Cohesion: 0.4
Nodes (6): <debug> library, Topic 12: Debug Library, cerebras.sdk.debug.debug_util.read_trace, Topic 13: Simprint Library, <simprint> library, Simprint flushes on newline to debug stalling programs

### Community 22 - "Sparse Tensors Tutorial"
Cohesion: 0.4
Nodes (5): topic-04 commands_wse3.sh, topic-04 layout.csl, topic-04 pe_program.csl, topic-04 run.py, Tutorial 04: Sparse Tensors

### Community 23 - "Sentinels Tutorial"
Cohesion: 0.4
Nodes (5): topic-05 layout.csl, topic-05 pe_program.csl, topic-05 run.py, topic-05 sentinel.csl, Tutorial 05: Sentinels

### Community 24 - "Filters Tutorial"
Cohesion: 0.4
Nodes (5): Tutorial 08: Filters, topic-08 layout.csl, topic-08 recv.csl, topic-08 run.py, topic-08 send.csl

### Community 25 - "Color Swap Tutorial"
Cohesion: 0.4
Nodes (5): Tutorial 14: Color Swap, topic-14 commands_wse2.sh, topic-14 layout.csl, topic-14 pe_program.csl, topic-14 run.py

### Community 26 - "WSE-3 Microthreads Tutorial"
Cohesion: 0.4
Nodes (5): topic-15 layout.csl, topic-15 left_pe.csl, topic-15 right_pe.csl, topic-15 run.py, Tutorial 15: WSE-3 Microthreads

### Community 27 - "FIFOs Tutorial"
Cohesion: 0.5
Nodes (4): topic-09 buffer.csl, Tutorial 09: FIFOs, topic-09 layout.csl, topic-09 run.py

### Community 28 - "Map Builtin Tutorial"
Cohesion: 0.5
Nodes (4): topic-10 layout.csl, Tutorial 10: Map Builtin, topic-10 pe_program.csl, topic-10 run.py

### Community 29 - "Residual Benchmark"
Cohesion: 0.5
Nodes (4): Residual |b - A*x|, 2x2 PE rectangle with memcpy infrastructure, GEMV / AXPY / NRMINF import_module, Rationale: SIMD used in GEMV/AXPY to reduce address-computation overhead

### Community 30 - "Row/Col Broadcast Benchmark"
Cohesion: 0.5
Nodes (4): memcpy_h2d_colbcast() API, memcpy_h2d_rowbcast() API, Host-to-Device Row/Column Broadcast Test, Rationale: avoids 3x host bandwidth waste of duplicating broadcast data

### Community 31 - "Wavelet Payload Filtering"
Cohesion: 0.5
Nodes (4): Topic 4: Wavelets for Sparse Tensors, Wavelet 32-bit payload upper/lower 16-bit split, Topic 8: Filters, Range filters on upper 16 bits

### Community 32 - "Game of Life Benchmark"
Cohesion: 0.67
Nodes (3): Benchmark: Game of Life, Cellular automaton (2D), Zero boundary conditions

### Community 33 - "Wide Multiplication Benchmark"
Cohesion: 0.67
Nodes (3): Concatenation of X and Y into single input vector, Rationale: single color used for both X and Y via concatenation, Wide Multiplication (128-bit x 128-bit = 256-bit)

### Community 34 - "Collectives Library & MPI"
Cohesion: 0.67
Nodes (3): Topic 11: Collective Communications, <collectives_2d> library, Message Passing Interface (MPI)

### Community 35 - "@map Builtin"
Cohesion: 1.0
Nodes (2): Topic 10: @map Builtin, @map leverages DSDs to avoid explicit loops and boost performance

### Community 36 - "Color Swap Routing"
Cohesion: 1.0
Nodes (2): Topic 14: Color Swap, swap_color_x routing option (WSE-2)

### Community 37 - "Microthread Queue Decoupling"
Cohesion: 1.0
Nodes (2): Decoupling microthread from queue ID conserves resources, Topic 13: WSE-3 Microthreads

### Community 38 - "Bandwidth Test"
Cohesion: 1.0
Nodes (1): Benchmark: Bandwidth Test

### Community 39 - "FFT 3D"
Cohesion: 1.0
Nodes (1): Benchmark: FFT 3D

### Community 40 - "Row-Col Broadcast"
Cohesion: 1.0
Nodes (1): Benchmark: Row-Col Broadcast

### Community 41 - "GEMV-06 Exercises"
Cohesion: 1.0
Nodes (1): GEMV-06 Exercises

## Knowledge Gaps
- **330 isolated node(s):** `Cerebras Wafer-Scale Cluster (WSC)`, `Appliance Logging (cerebras.appliance.logger)`, `csctl CLI tool for job monitoring`, `Rationale: resource_cpu/resource_mem kwargs to reduce queue wait`, `Task ID` (+325 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **Thin community `@map Builtin`** (2 nodes): `Topic 10: @map Builtin`, `@map leverages DSDs to avoid explicit loops and boost performance`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Color Swap Routing`** (2 nodes): `Topic 14: Color Swap`, `swap_color_x routing option (WSE-2)`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Microthread Queue Decoupling`** (2 nodes): `Decoupling microthread from queue ID conserves resources`, `Topic 13: WSE-3 Microthreads`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Bandwidth Test`** (1 nodes): `Benchmark: Bandwidth Test`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `FFT 3D`** (1 nodes): `Benchmark: FFT 3D`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Row-Col Broadcast`** (1 nodes): `Benchmark: Row-Col Broadcast`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `GEMV-06 Exercises`** (1 nodes): `GEMV-06 Exercises`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `CSL Code Examples Index` connect `Tutorial & Example Concepts` to `DSDs & Memory Operations`, `Pipelines, DSRs & Mandelbrot`?**
  _High betweenness centrality (0.174) - this node is a cross-community bridge._
- **Why does `fabin_dsd / fabout_dsd` connect `Pipelines, DSRs & Mandelbrot` to `Tasks, Colors & Builtins`, `Tutorial & Example Concepts`?**
  _High betweenness centrality (0.126) - this node is a cross-community bridge._
- **Why does `Tutorial GEMV 02: Memory DSDs` connect `DSDs & Memory Operations` to `Tutorial & Example Concepts`?**
  _High betweenness centrality (0.106) - this node is a cross-community bridge._
- **What connects `Cerebras Wafer-Scale Cluster (WSC)`, `Appliance Logging (cerebras.appliance.logger)`, `csctl CLI tool for job monitoring` to the rest of the system?**
  _330 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Tutorial & Example Concepts` be split into smaller, more focused modules?**
  _Cohesion score 0.03 - nodes in this community are weakly interconnected._
- **Should `Appliance & Runtime Infrastructure` be split into smaller, more focused modules?**
  _Cohesion score 0.05 - nodes in this community are weakly interconnected._
- **Should `Compiler Flags & Builtins` be split into smaller, more focused modules?**
  _Cohesion score 0.05 - nodes in this community are weakly interconnected._