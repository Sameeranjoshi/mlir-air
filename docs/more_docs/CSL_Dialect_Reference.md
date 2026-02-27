# CSL Dialect Reference

## 1. Operations Summary

# CSL Dialect Ops Summary

This document summarizes the current design of the CSL dialect operations in MLIR, which have been aligned with the spatial placement/routing programming model. The operations model Cerebras CSL concepts like spatial code regions, routing, coloring, and dataflow.

## 1. Type System

*   `!csl.color` - Represents a CSL communication color symbol/ID.
*   `!csl.code_region` - Represents a local block of PE grid logic (shape, body) that hasn't been globally placed on the chip yet.
*   `!csl.port` - A typed communication endpoint attached to a `!csl.code_region`, optionally configured with edge/route.
*   `!csl.kernel` - A handle pointing to a specific CSL source file kernel and its set of static compilation parameters.

## 2. Global Container

#### `csl.spatial_placement`
The top-level container for a full CSL layout and placement program. It defines the global context for colors, routes, kernels, port connections, and places `csl.code_region` sub-regions onto absolute WSE compute fabric coordinates.

```mlir
csl.spatial_placement {
  // global definitions
}
```

## 3. Communication Primitives (Routing and Colors)
*(Defined in `CSLRoutingOps.td`)*

#### `csl.color`
Declares a color symbol. The compiler automatically maps these to physical HW colors (`[0, 24]`) if not explicitly specified.
`%c0 = csl.color : !csl.color` (compiler managed)
`%c1 = csl.color 0 : !csl.color` (hardcoded physical color 0)

#### `csl.route`
Declares an input-to-output mapping for wavelets using the directions: `NORTH`, `SOUTH`, `EAST`, `WEST`, and `RAMP` (local compute). This produces an `i32` identifier handle for subsequent ops.
`%r1 = csl.route in(RAMP) out(EAST) : i32`
`%r2 = csl.route in(WEST) out(RAMP) : i32`

## 4. Kernels
*(Defined in `CSLKernelOps.td`)*

#### `csl.kernel`
Points to a CSL code file, optional COMPILE-time parameters (e.g. `col`, `row`), and contains metadata of symbols like parameters (`csl.param`), variables (`csl.var`), and functions (`csl.func`).
```mlir
%k1 = csl.kernel "pe_program.csl" params({param1 = 100 : i32}) {
  csl.func @compute() { csl.return }
} : !csl.kernel
```

## 5. Layout and Placement Ops
*(Defined in `CSLLayoutOps.td`)*

#### `csl.code_region`
Defines a reusable code region with a given layout shape (width/height), a suite of accepted colors/routes, and an inner structure modeled via a body containing `csl.paint` operations to color/route individual PEs. Note that it is not placed at absolute coordinates yet.
```mlir
%reg = csl.code_region routes(%r1) colors(%c) shape(10, 10) {
  // inner configuration of the 10x10 tile
} : !csl.code_region
```

#### `csl.paint` (Body Content of `csl.code_region`)
Inside the `csl.code_region` body, this assigns specific routes and colors to the local coordinate of the PE within that region.
```mlir
csl.paint pe(0, 0) route(%r1) color(%c)
```

#### `csl.port`
Declares a physical communication port for a region, indicating data direction ("input" or "output"), the route configuration it will use, and size in element count.
```mlir
%port1 = csl.port %reg type("output") route(%r1) size(256) : !csl.port
```

#### `csl.place`
Binds a kernel (CSL code file/parameters) to a defined `csl.code_region` layout and places them at an absolute global `(x, y)` coordinate on the WSE computing fabric.
```mlir
csl.place %reg at(0, 0) kernel(%k1)
```

#### `csl.dataflow`
Connects an output port of a region to an input port of another region, establishing cross-region communication. The compiler (like `aie.flow()`) creates optimal pathways for the required color between the bounding boxes.
```mlir
csl.dataflow %src_port -> %dst_port
```

---
*Note: SdkLayout-specific ops matching the Python API (like `sym_color`, `paint_all`, `create_output_port` etc.) are currently disabled and commented out via `#ifdef CSL_ENABLE_SDK_LAYOUT_OPS` in the standard layout ops definition file, to purely favor the simpler dataflow/spatial placement semantic definitions.*


---

## 2. Comprehensive CSL Dialect

# CSL Dialect Reference

The `csl` dialect provides a structured MLIR representation of Cerebras Systems Language (CSL) programs targeting the Wafer-Scale Engine (WSE). It mirrors the key CSL constructs — layout blocks, PE tile assignment, color-based routing, DSD-driven data movement, task-based kernels, and host runtime interfaces — as first-class MLIR operations with verification and type safety.

This dialect is part of **Phase 2** of the [design plan](design_air_to_csl.md) (Section 9.2): building `csl.*` MLIR ops that mirror CSL language constructs, enabling MLIR-level verification and optimization before text emission via `air-translate --csl-emit`.

---

## Table of Contents

1. [Motivation](#1-motivation)
2. [Dialect Architecture](#2-dialect-architecture)
3. [Types](#3-types)
4. [Enumerations](#4-enumerations)
5. [Operations by Category](#5-operations-by-category)
   - [5.1 Layout](#51-layout)
   - [5.2 Placement](#52-placement)
   - [5.3 Routing](#53-routing)
   - [5.4 Kernel](#54-kernel)
   - [5.5 Data Movement](#55-data-movement)
   - [5.6 Runtime](#56-runtime)
6. [Building a Complete Program](#6-building-a-complete-program)
7. [Testing](#7-testing)
8. [Build System Integration](#8-build-system-integration)
9. [Design Rationale](#9-design-rationale)

---

## 1. Motivation

The Phase 1 text emitter (`AIRToCSLPass.cpp`) walks AIR IR and directly emits CSL source files. While functional, it has fundamental limitations:

- No MLIR-level verification of the generated code structure.
- No composability with optimization passes between AIR lowering and text emission.
- The emitter is a long `dyn_cast` chain that grows unmanageably as more CSL constructs are needed.
- Cannot represent or optimize inter-PE communication at the IR level.

The CSL dialect addresses these by representing CSL constructs as proper MLIR operations that participate in MLIR's verification, transformation, and analysis infrastructure.

---

## 2. Dialect Architecture

All operations share the single `csl` MLIR dialect namespace (`xilinx::csl` in C++) but are organized into six **conceptual sub-categories** across separate TableGen files for clarity:

| Category | TableGen File | Purpose |
|---|---|---|
| Layout | `CSLLayoutOps.td` | Spatial grid configuration |
| Placement | `CSLPlacementOps.td` | Tile-to-code mapping |
| Routing | `CSLRoutingOps.td` | Color declaration and routing |
| Kernel | `CSLKernelOps.td` | PE-level compute (functions, tasks, variables) |
| Data Movement | `CSLDataMovementOps.td` | DSD creation and bulk moves |
| Runtime | `CSLRuntimeOps.td` | Module system, exports, comptime blocks |

Shared definitions (the dialect itself, base op class, types, enums) live in `CSLBase.td`. All sub-category files include `CSLBase.td` and are aggregated by `CSLOps.td`.

### Directory Structure

```
mlir/include/air/Dialect/CSL/
├── CSLBase.td              # Dialect, types, enums
├── CSLOps.td               # Master include (aggregates all .td files)
├── CSLLayoutOps.td         # csl.layout, csl.set_rectangle
├── CSLPlacementOps.td      # csl.set_tile_code
├── CSLRoutingOps.td        # csl.color, csl.route
├── CSLKernelOps.td         # csl.func, csl.task, csl.return, csl.var
├── CSLDataMovementOps.td   # csl.get_mem_dsd, csl.get_fab_dsd, csl.mov
├── CSLRuntimeOps.td        # csl.import_module, csl.export_name, csl.export_symbol,
│                           # csl.comptime, csl.module, csl.param
├── CSLDialect.h            # C++ type declarations
├── CSLOps.h                # C++ op header
└── CMakeLists.txt          # TableGen generation rules

mlir/lib/Dialect/CSL/IR/
├── CSLDialect.cpp          # Dialect initialization, type parsing/printing
├── CSLOps.cpp              # Custom parsers/printers for csl.func and csl.task
└── CMakeLists.txt          # Library build rules

mlir/test/Dialect/CSL/
├── layout_ops.mlir         # Layout op tests
├── placement_ops.mlir      # Placement op tests
├── routing_ops.mlir        # Routing op tests
├── kernel_ops.mlir         # Kernel op tests
├── datamovement_ops.mlir   # Data movement op tests
├── runtime_ops.mlir        # Runtime op tests
├── complete_program.mlir   # Full program integration test
├── roundtrip.mlir          # Parse-print-parse consistency test
├── invalid.mlir            # Verifier diagnostic tests
└── invalid_parse.mlir      # Parser rejection tests
```

---

## 3. Types

The dialect defines three custom types, all declared in C++ and referenced via `DialectType` predicates in TableGen.

| Type | MLIR Syntax | Description |
|---|---|---|
| `ColorType` | `!csl.color` | A WSE communication color handle. Colors identify logical channels on the fabric. |
| `DsdType` | `!csl.dsd` | A Data Structure Descriptor. DSDs describe patterns of memory or fabric access. |
| `ImportedModuleType` | `!csl.imported_module` | A handle to an imported CSL library module. |

### Parsing

Types are parsed/printed via `CSLDialect::parseType` and `CSLDialect::printType` using keyword matching:

```
!csl.color           → ColorType
!csl.dsd             → DsdType
!csl.imported_module → ImportedModuleType
```

---

## 4. Enumerations

Two `I32EnumAttr` enumerations are defined in `CSLBase.td`:

### `Direction` (routing directions)

```
NORTH = 0
SOUTH = 1
EAST  = 2
WEST  = 3
RAMP  = 4
```

`RAMP` represents the connection to the host/off-chip memory, a unique WSE concept where the "ramp" lanes connect PE tiles to external DRAM.

### `DsdKind` (DSD categories)

```
mem1d  = 0   # 1D contiguous memory DSD
mem2d  = 1   # 2D strided memory DSD (future)
fabin  = 2   # Fabric input DSD (receive from network)
fabout = 3   # Fabric output DSD (send to network)
```

---

## 5. Operations by Category

### 5.1 Layout

#### `csl.layout`

Top-level container for spatial configuration. Corresponds to the `layout { }` block in CSL, which is executed on the host to configure the PE grid before program launch.

**Traits:** `SingleBlock`, `NoTerminator`

```mlir
csl.layout {
  csl.set_rectangle 2, 2
  csl.set_tile_code 0, 0 file("pe_program.csl")
  csl.export_name "data" : f32
}
```

**Rationale:** CSL separates layout (host-side spatial configuration) from PE code. The `csl.layout` region captures this separation at the IR level, constraining spatial-configuration ops to a well-defined scope.

#### `csl.set_rectangle`

Sets the rectangular PE grid dimensions. Corresponds to `@set_rectangle(width, height)` in CSL.

**Arguments:** `I64Attr:$width`, `I64Attr:$height`

```mlir
csl.set_rectangle 4, 8    // 4 columns, 8 rows
csl.set_rectangle 750, 994 // WSE-3 maximum
```

### 5.2 Placement

#### `csl.set_tile_code`

Assigns a CSL source module to a PE at grid position (x, y). Corresponds to `@set_tile_code(x, y, "file.csl", .{ params })` in CSL.

**Arguments:** `I64Attr:$x`, `I64Attr:$y`, `StrAttr:$file`, `OptionalAttr<DictionaryAttr>:$params`

```mlir
// Without params — all tiles run the same code
csl.set_tile_code 0, 0 file("pe_program.csl")

// With compile-time params — per-tile specialization
csl.set_tile_code 1, 1 file("pe_program.csl") params({col = 1 : i32, row = 1 : i32})
```

**Rationale:** The optional `params` dictionary mirrors CSL's compile-time parameter passing, enabling per-tile specialization without duplicating modules. This is a key difference from AIE, where code is cloned per tile.

### 5.3 Routing

#### `csl.color`

Declares a communication color with a given integer ID. Colors are the fundamental routing primitive on the WSE fabric — each color identifies a logical channel through which wavelets (data packets) travel.

**Arguments:** `I32Attr:$id`
**Results:** `!csl.color`

```mlir
%c0 = csl.color 0 : !csl.color
%c1 = csl.color 1 : !csl.color
```

#### `csl.route`

Binds a color to a fabric direction. Determines how wavelets tagged with this color propagate through the on-chip network.

**Arguments:** `CSL_ColorType:$color`, `CSL_DirectionEnum:$direction`

```mlir
%c = csl.color 0 : !csl.color
csl.route %c dir(EAST)    // Wavelets on color 0 travel east
csl.route %c dir(RAMP)    // Color 0 also connects to host DRAM
```

**Rationale:** CSL's routing model is color-centric: you declare colors, then bind them to directions. This is fundamentally different from AIE's circuit-switched flows or TT's NoC multicast. Representing colors as SSA values enables dataflow analysis of the communication graph.

### 5.4 Kernel

#### `csl.func`

Defines a CSL function on a PE. The body contains standard MLIR operations (`arith`, `scf`, `memref`) representing the compute kernel. Corresponds to `fn name() void { ... }` in CSL.

**Traits:** `IsolatedFromAbove`, `Symbol`

```mlir
csl.func @compute() {
  // Standard MLIR compute ops go here
  csl.return
}
```

**Custom assembly format:** The parser and printer are hand-written in `CSLOps.cpp` to produce clean `csl.func @name() { ... }` syntax rather than the generic op format.

#### `csl.task`

Defines an event-driven task on a PE. Tasks are triggered when a wavelet arrives on the bound color. This is the fundamental execution mechanism on the WSE.

**Traits:** `IsolatedFromAbove`, `Symbol`
**Arguments:** `SymbolNameAttr:$sym_name`, `I32Attr:$color_id`

```mlir
csl.task @recv_data() color(3) {
  csl.return
}
```

**Rationale:** Tasks are unique to WSE's event-driven execution model. Unlike AIE (sequential within a core) or TT (kernel-launch-based), WSE tasks are activated by wavelet arrival on a specific color. The `color(N)` syntax makes this binding explicit.

#### `csl.return`

Terminator for `csl.func` and `csl.task` regions.

**Traits:** `Terminator`, `ParentOneOf<["FuncOp", "TaskOp"]>`

```mlir
csl.func @f() {
  csl.return    // required terminator
}
```

#### `csl.var`

Declares a PE-local variable. Corresponds to `var name: [N]T` in CSL.

**Traits:** `Symbol`
**Arguments:** `SymbolNameAttr:$sym_name`, `TypeAttr:$type`

```mlir
csl.var @buf : memref<1024xf32>
csl.var @matrix : memref<32x32xf16>
csl.var @scalar : f32
```

### 5.5 Data Movement

#### `csl.get_mem_dsd`

Creates a memory-backed Data Structure Descriptor referencing a contiguous region of PE-local memory. Corresponds to `@get_dsd(mem1d_dsd, ...)` in CSL.

**Arguments:** `AnyMemRef:$buffer`, `Index:$length`
**Results:** `!csl.dsd`

```mlir
%dsd = csl.get_mem_dsd %buf, %len : memref<1024xf32>, index -> !csl.dsd
```

#### `csl.get_fab_dsd`

Creates a fabric-backed DSD. Used for streaming data to/from neighboring PEs via colors.

**Arguments:** `CSL_DsdKindEnum:$kind`, `CSL_ColorType:$color`, `Index:$length`
**Results:** `!csl.dsd`

```mlir
%c = csl.color 0 : !csl.color
%dsd_in  = csl.get_fab_dsd fabin  %c, %len : !csl.color, index -> !csl.dsd
%dsd_out = csl.get_fab_dsd fabout %c, %len : !csl.color, index -> !csl.dsd
```

**Rationale:** DSDs are WSE's primary mechanism for efficient bulk data access. By representing them as SSA values in MLIR, optimization passes can analyze and transform DSD patterns — for example, converting scalar loops into bulk DSD moves, or merging compatible DSDs.

#### `csl.mov`

Performs a hardware-accelerated bulk data move between two DSDs. Corresponds to `@mov32` or `@mov16` builtins in CSL.

**Arguments:** `CSL_DsdType:$dst`, `CSL_DsdType:$src`

```mlir
csl.mov %dst_dsd, %src_dsd : !csl.dsd, !csl.dsd
```

### 5.6 Runtime

#### `csl.import_module`

Imports a CSL library module by name, optionally passing compile-time parameters. Corresponds to `@import_module("name", params)` in CSL.

**Arguments:** `StrAttr:$module_name`, `OptionalAttr<DictionaryAttr>:$params`
**Results:** `!csl.imported_module`

```mlir
%mod = csl.import_module "<memcpy/memcpy>" : !csl.imported_module
%mod2 = csl.import_module "<memcpy/get_params>" params({width = 2 : i32, height = 2 : i32}) : !csl.imported_module
```

#### `csl.export_name`

Declares a host-visible symbol in the layout. Used inside `csl.layout` blocks to make PE data and entry points accessible via the `SdkRuntime` host API. Corresponds to `@export_name("name", type, is_ptr)` in CSL.

**Arguments:** `StrAttr:$sym_name`, `TypeAttr:$type`

```mlir
csl.export_name "data" : f32
csl.export_name "compute" : () -> ()
csl.export_name "buffer" : memref<1024xf32>
```

#### `csl.export_symbol`

Exports a symbol from a PE module. Used inside `csl.comptime` blocks to make symbols visible to the layout or host. Corresponds to `@export_symbol(sym, "alias")` in CSL.

**Arguments:** `FlatSymbolRefAttr:$sym`, `OptionalAttr<StrAttr>:$alias`

```mlir
csl.export_symbol @compute
csl.export_symbol @buf_ptr alias("data")
```

#### `csl.comptime`

A region evaluated at compile time by the CSL compiler. Used for binding tasks to colors, exporting symbols, and other static configuration. Corresponds to `comptime { ... }` in CSL.

**Traits:** `SingleBlock`, `NoTerminator`

```mlir
csl.comptime {
  csl.export_symbol @compute
  csl.export_symbol @buf alias("buffer")
}
```

#### `csl.module`

Defines a CSL module (PE program) that can be assigned to PE tiles. Contains parameter declarations, variable declarations, functions, tasks, and comptime blocks.

**Traits:** `IsolatedFromAbove`, `Symbol`, `SymbolTable`, `SingleBlock`, `NoTerminator`

```mlir
csl.module @pe_program {
  csl.param @tile_id : i32
  csl.var @buf : memref<1024xf32>
  csl.func @compute() { csl.return }
  csl.comptime { csl.export_symbol @compute }
}
```

**Rationale:** `csl.module` has both `Symbol` (so it can be referenced by name from `csl.set_tile_code`) and `SymbolTable` (so it can contain named symbols like `csl.var`, `csl.func`, `csl.param`). `IsolatedFromAbove` enforces that PE code cannot reference values from the layout scope — matching CSL's strong separation between host-side layout code and PE-side kernel code.

#### `csl.param`

Declares a compile-time module parameter. Parameters are specialized per-tile when assigned via `csl.set_tile_code`. Corresponds to `param name: type;` in CSL.

**Traits:** `Symbol`
**Arguments:** `SymbolNameAttr:$sym_name`, `TypeAttr:$type`

```mlir
csl.param @memcpy_params : i64
csl.param @tile_id : i32
```

---

## 6. Building a Complete Program

A CSL program in MLIR follows a two-part structure that mirrors real CSL code: a **layout** section (host-side spatial configuration) and one or more **module** definitions (PE-side code).

### Step 1: Define the PE grid

The `csl.layout` block sets up the grid and assigns code to tiles:

```mlir
csl.layout {
  csl.set_rectangle 2, 2
  csl.set_tile_code 0, 0 file("pe_program.csl")
  csl.set_tile_code 1, 0 file("pe_program.csl")
  csl.set_tile_code 0, 1 file("pe_program.csl")
  csl.set_tile_code 1, 1 file("pe_program.csl")
```

### Step 2: Export host-visible symbols

Still inside the layout, declare symbols the host can access:

```mlir
  csl.export_name "arg_0" : memref<24xf32>
  csl.export_name "init_and_compute" : () -> ()
}
```

### Step 3: Define the PE module

The `csl.module` contains all PE-level declarations:

```mlir
csl.module @pe_program {
  csl.param @memcpy_params : i64          // compile-time params
  csl.var @arg_0 : memref<24xf32>         // PE-local memory
```

### Step 4: Write compute functions and tasks

```mlir
  csl.func @compute() {
    // Standard MLIR ops (arith, scf, memref) go here
    csl.return
  }
```

### Step 5: Export symbols at compile time

The `csl.comptime` block binds PE-internal symbols to host-visible names:

```mlir
  csl.comptime {
    csl.export_symbol @arg_0 alias("arg_0")
    csl.export_symbol @init_and_compute
  }
}
```

### Complete Example

Putting it all together — a GEMV kernel on a 2x2 grid:

```mlir
csl.layout {
  csl.set_rectangle 2, 2
  csl.set_tile_code 0, 0 file("pe_program.csl")
  csl.set_tile_code 1, 0 file("pe_program.csl")
  csl.set_tile_code 0, 1 file("pe_program.csl")
  csl.set_tile_code 1, 1 file("pe_program.csl")
  csl.export_name "arg_0" : memref<24xf32>
  csl.export_name "arg_1" : memref<6xf32>
  csl.export_name "arg_2" : memref<4xf32>
  csl.export_name "init_and_compute" : () -> ()
}

csl.module @pe_program {
  csl.param @memcpy_params : i64
  csl.var @arg_0 : memref<24xf32>
  csl.var @arg_1 : memref<6xf32>
  csl.var @arg_2 : memref<4xf32>

  csl.func @compute() {
    csl.return
  }

  csl.func @init_and_compute() {
    csl.return
  }

  csl.comptime {
    csl.export_symbol @arg_0 alias("arg_0")
    csl.export_symbol @arg_1 alias("arg_1")
    csl.export_symbol @arg_2 alias("arg_2")
    csl.export_symbol @init_and_compute
  }
}
```

### With Routing and Data Movement

Adding inter-PE communication:

```mlir
// Routing configuration
%send_color = csl.color 0 : !csl.color
%ack_color  = csl.color 1 : !csl.color
csl.route %send_color dir(EAST)
csl.route %ack_color dir(WEST)

// Data movement inside a function
func.func @transfer(%buf : memref<256xf32>) {
  %len = arith.constant 256 : index
  %c = csl.color 0 : !csl.color

  %mem_dsd = csl.get_mem_dsd %buf, %len : memref<256xf32>, index -> !csl.dsd
  %fab_out = csl.get_fab_dsd fabout %c, %len : !csl.color, index -> !csl.dsd

  csl.mov %fab_out, %mem_dsd : !csl.dsd, !csl.dsd
  return
}
```

---

## 7. Testing

Tests live in `mlir/test/Dialect/CSL/` and use the MLIR FileCheck framework. The lit test target `check-airmlir-dialect-csl` runs all CSL dialect tests.

### Running Tests

```bash
# Run all CSL dialect tests via lit (from the build directory)
cd build && lit --config-prefix=lit.site mlir/test/Dialect/CSL/ -v

# Run as part of the full test suite
cd build && lit --config-prefix=lit.site mlir/test/ -v
```

### Test Categories

| File | Category | Description |
|---|---|---|
| `layout_ops.mlir` | Positive | Grid dimensions, single PE, WSE-3 max, composite layout |
| `placement_ops.mlir` | Positive | Tile code assignment, params, different modules per tile |
| `routing_ops.mlir` | Positive | All five directions, multiple colors, high color IDs |
| `kernel_ops.mlir` | Positive | Variables, functions, tasks with various color IDs |
| `datamovement_ops.mlir` | Positive | Memory/fabric DSDs, bulk moves, multi-color pipelines |
| `runtime_ops.mlir` | Positive | Module imports, exports, params, comptime blocks |
| `complete_program.mlir` | Integration | Full GEMV-like program with all six categories |
| `roundtrip.mlir` | Roundtrip | `--verify-roundtrip` parse-print-parse consistency |
| `invalid.mlir` | Negative | Verifier rejects misplaced ops (wrong parent, missing traits) |
| `invalid_parse.mlir` | Negative | Parser rejects malformed syntax (wrong keywords, bad enums) |

---

## 8. Build System Integration

### CMake Targets

- **`CSLDialect`** — Static library containing the dialect implementation.
- **`csl-headers`** — Generated TableGen headers.
- **`check-airmlir-dialect-csl`** — Lit test target for CSL dialect tests.

### Dependencies

The CSL dialect depends on:
- MLIR core (`MLIRIR`, `MLIRSupport`, `MLIRInferTypeOpInterface`)
- MLIR dialect interfaces (`MLIRCallInterfaces`, `MLIRSideEffectInterfaces`)

It is registered in `mlir/lib/InitAll.cpp` alongside the AIR and AIRRt dialects, so `air-opt` automatically recognizes `csl.*` operations.

---

## 9. Design Rationale

### Single Dialect, Multiple TableGen Files

Rather than creating six separate MLIR dialects (`csl_layout`, `csl_routing`, etc.), all operations share one `csl` dialect namespace. This avoids the overhead of six dialect registrations, six C++ namespaces, and cross-dialect type/attribute sharing boilerplate. The six conceptual categories are separated via TableGen files for maintainability.

### Custom Types vs. Attributes

Colors, DSDs, and imported modules are represented as MLIR types (not attributes) because they flow through SSA dataflow — a `csl.color` produces a `!csl.color` value consumed by `csl.route` and `csl.get_fab_dsd`. This enables standard MLIR dataflow analysis.

### `IsolatedFromAbove` on Module/Func/Task

The `IsolatedFromAbove` trait on `csl.module`, `csl.func`, and `csl.task` mirrors CSL's strong scoping rules: PE code cannot reference layout-level values, and function bodies cannot capture variables from their enclosing scope. This enables safe parallel compilation and separate code generation.

### `SymbolTable` on `csl.module`

`csl.module` carries both `Symbol` (to be referenced by name) and `SymbolTable` (to contain named symbols). This is necessary because `csl.var`, `csl.param`, `csl.func`, and `csl.task` all carry the `Symbol` trait, which requires their parent to have `SymbolTable`.

### `SingleBlock` + `NoTerminator` on Region Ops

`csl.layout`, `csl.comptime`, and `csl.module` use `SingleBlock` + `NoTerminator` because they represent declarative regions (configuration blocks) rather than control-flow regions. There is no branching or early return within a layout or comptime block.

### Custom Parsers for `csl.func` and `csl.task`

These ops use hand-written parsers/printers (in `CSLOps.cpp`) instead of declarative `assemblyFormat` because:
1. `csl.func @name() { ... }` syntax is cleaner than what the generic format would produce.
2. `csl.task @name() color(N) { ... }` requires custom parsing of the `color(N)` suffix after the argument list.

All other ops use declarative `assemblyFormat` defined in TableGen, which is preferred for maintainability.

### Relationship to AIR Dialect

The CSL dialect is a **lowering target** for AIR. The `air-opt --air-to-csl` pass (under development) transforms AIR operations into CSL operations:

```
air.launch       → csl.layout + csl.set_rectangle
air.segment      → csl.set_tile_code (for each PE)
air.herd body    → csl.module with csl.func/csl.task
air.channel.*    → csl.color + csl.route + csl.get_fab_dsd + csl.mov
```

The CSL dialect is then consumed by `air-translate --csl-emit`, which walks the `csl.*` ops and emits syntactically valid CSL source files (`layout.csl`, `pe_program.csl`, `run.py`).


---

## 3. CSL Dialect Implementation Details

<!-- Autogenerated by mlir-tblgen; don't manually edit -->

# 'csl' Dialect

The `csl` dialect provides a structured MLIR representation of Cerebras
Systems Language (CSL) programs targeting Wafer-Scale Engine (WSE)
hardware. It mirrors key CSL constructs -- layout blocks, PE tile
assignment, color-based routing, DSD-driven data movement, task-based
kernels, and host runtime interfaces -- as first-class MLIR operations
with verification and optimization opportunities.

The dialect is organized into six conceptual sub-dialect categories,
all sharing the `csl` namespace:

- **Layout**: top-level spatial configuration (`csl.layout`,
  `csl.set_rectangle`).
- **Placement**: tile-to-code mapping (`csl.set_tile_code`).
- **Routing**: color declaration and directional routing
  (`csl.color`, `csl.route`).
- **Kernel**: PE-level compute (`csl.func`, `csl.task`, `csl.var`).
- **Data Movement**: DSD creation and bulk moves (`csl.get_mem_dsd`,
  `csl.get_fab_dsd`, `csl.mov`).
- **Runtime**: module system and host interface
  (`csl.import_module`, `csl.export_name`, `csl.export_symbol`,
   `csl.module`, `csl.comptime`).

[TOC]

## Operations

### `csl.code_region` (xilinx::csl::CodeRegionOp)

_Create a CodeRegion from a CSL source file_

Syntax:

```
operation ::= `csl.code_region` $source_file `name` `(` $region_name `)` `width` `(` $width `)` `height` `(` $height `)` attr-dict `:` type($result)
```

Corresponds to `layout.create_code_region(source, name, width, height)`.
Returns an opaque handle used by painting, param, port, and placement ops.

Example:
```mlir
%r = csl.code_region "./pe_program.csl" name("pe") width(4) height(1)
```

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>source_file</code></td><td>::mlir::StringAttr</td><td>string attribute</td></tr>
<tr><td><code>region_name</code></td><td>::mlir::StringAttr</td><td>string attribute</td></tr>
<tr><td><code>width</code></td><td>::mlir::IntegerAttr</td><td>64-bit signless integer attribute</td></tr>
<tr><td><code>height</code></td><td>::mlir::IntegerAttr</td><td>64-bit signless integer attribute</td></tr>
</table>

#### Results:

| Result | Description |
| :----: | ----------- |
| `result` | CSL code region handle |



### `csl.color` (xilinx::csl::ColorOp)

_Declare a communication color_

Syntax:

```
operation ::= `csl.color` $id attr-dict `:` type($result)
```

Declares a CSL communication color with the given integer ID.
Colors are the fundamental routing primitive on the WSE fabric.

Example:
```mlir
%c0 = csl.color 0 : !csl.color
%c1 = csl.color 1 : !csl.color
```

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>id</code></td><td>::mlir::IntegerAttr</td><td>32-bit signless integer attribute</td></tr>
</table>

#### Results:

| Result | Description |
| :----: | ----------- |
| `result` | CSL color type |



### `csl.comptime` (xilinx::csl::ComptimeOp)

_Compile-time evaluation block_

Syntax:

```
operation ::= `csl.comptime` $body attr-dict
```

A region evaluated at compile time by the CSL compiler. Used for
binding tasks to colors, exporting symbols, and other static
configuration. Corresponds to `comptime { ... }` in CSL.

Example:
```mlir
csl.comptime {
  csl.export_symbol @compute
}
```

Traits: `NoTerminator`, `SingleBlock`



### `csl.connect` (xilinx::csl::ConnectOp)

_Connect an output port to an input port_

Syntax:

```
operation ::= `csl.connect` $tx `->` $rx attr-dict
```

Corresponds to `layout.connect(tx_port, rx_port)`. The compiler finds
the routing path between the two ports automatically.

Example:
```mlir
csl.connect %tx_port -> %rx_port
```

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `tx` | CSL port handle |
| `rx` | CSL port handle |



### `csl.create_input_port` (xilinx::csl::CreateInputPortOp)

_Create an input port on a region edge_

Syntax:

```
operation ::= `csl.create_input_port` $region `color` `(` $color `)` `edge` `(` $edge `)` `routing` `(` $routing_positions `)`
              `size` `(` $data_size `)` attr-dict `:` type($result)
```

Corresponds to:
  `region.create_input_port(color, edge, [routing_positions], data_size)`.

For input ports the routing positions should only specify output into RAMP
(data flows from fabric into the PE compute unit).

Example:
```mlir
%rp = csl.routing_position inputs([]) outputs([4])
%port = csl.create_input_port %r color(%c) edge(LEFT) routing(%rp) size(10)
```

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>edge</code></td><td>xilinx::csl::EdgeAttr</td><td>CSL port edge (SdkLayout)</td></tr>
<tr><td><code>data_size</code></td><td>::mlir::IntegerAttr</td><td>64-bit signless integer attribute</td></tr>
</table>

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `region` | CSL code region handle |
| `color` | CSL color type |
| `routing_positions` | variadic of 32-bit signless integer |

#### Results:

| Result | Description |
| :----: | ----------- |
| `result` | CSL port handle |



### `csl.create_input_stream` (xilinx::csl::CreateInputStreamOp)

_Create a host-to-device (H2D) I/O stream_

Syntax:

```
operation ::= `csl.create_input_stream` $port (`buf` `(` $io_buffer_size^ `)`)? attr-dict `:` type($result)
```

Corresponds to `layout.create_input_stream(rx_port, io_buffer_size=N)`.
Returns a stream handle used by the runtime to send data from the host.

Example:
```mlir
%s = csl.create_input_stream %rx_port : !csl.stream
%s = csl.create_input_stream %rx_port buf(1024) : !csl.stream
```

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>io_buffer_size</code></td><td>::mlir::IntegerAttr</td><td>64-bit signless integer attribute</td></tr>
</table>

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `port` | CSL port handle |

#### Results:

| Result | Description |
| :----: | ----------- |
| `result` | CSL stream handle |



### `csl.create_output_port` (xilinx::csl::CreateOutputPortOp)

_Create an output port on a region edge_

Syntax:

```
operation ::= `csl.create_output_port` $region `color` `(` $color `)` `edge` `(` $edge `)` `routing` `(` $routing_positions `)`
              `size` `(` $data_size `)` attr-dict `:` type($result)
```

Corresponds to:
  `region.create_output_port(color, edge, [routing_positions], data_size)`.

For output ports the routing positions must NOT specify an explicit
output direction — the compiler chooses the global routing path.

Edge values: LEFT=0, RIGHT=1, TOP=2, BOTTOM=3.

Example:
```mlir
%rp = csl.routing_position inputs([4]) outputs([])
%port = csl.create_output_port %r color(%c) edge(RIGHT) routing(%rp) size(10)
```

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>edge</code></td><td>xilinx::csl::EdgeAttr</td><td>CSL port edge (SdkLayout)</td></tr>
<tr><td><code>data_size</code></td><td>::mlir::IntegerAttr</td><td>64-bit signless integer attribute</td></tr>
</table>

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `region` | CSL code region handle |
| `color` | CSL color type |
| `routing_positions` | variadic of 32-bit signless integer |

#### Results:

| Result | Description |
| :----: | ----------- |
| `result` | CSL port handle |



### `csl.create_output_stream` (xilinx::csl::CreateOutputStreamOp)

_Create a device-to-host (D2H) I/O stream_

Syntax:

```
operation ::= `csl.create_output_stream` $port (`buf` `(` $io_buffer_size^ `)`)? attr-dict `:` type($result)
```

Corresponds to `layout.create_output_stream(tx_port, io_buffer_size=N)`.
Returns a stream handle used by the runtime to receive data from the device.

Example:
```mlir
%s = csl.create_output_stream %tx_port : !csl.stream
```

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>io_buffer_size</code></td><td>::mlir::IntegerAttr</td><td>64-bit signless integer attribute</td></tr>
</table>

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `port` | CSL port handle |

#### Results:

| Result | Description |
| :----: | ----------- |
| `result` | CSL stream handle |



### `csl.export_name` (xilinx::csl::ExportNameOp)

_Declare a host-visible symbol in the layout_

Syntax:

```
operation ::= `csl.export_name` $sym_name `:` $type attr-dict
```

Exports a named symbol to the host runtime, making it accessible
via `SdkRuntime` API calls. Used inside `csl.layout` blocks.
Corresponds to `@export_name("name", type, is_ptr)` in CSL.

Example:
```mlir
csl.export_name "data" : f32
csl.export_name "compute" : () -> ()
```

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>sym_name</code></td><td>::mlir::StringAttr</td><td>string attribute</td></tr>
<tr><td><code>type</code></td><td>::mlir::TypeAttr</td><td>any type attribute</td></tr>
</table>



### `csl.export_symbol` (xilinx::csl::ExportSymbolOp)

_Export a symbol from a PE module_

Syntax:

```
operation ::= `csl.export_symbol` $sym (`alias` `(` $alias^ `)`)? attr-dict
```

Exports a symbol from the PE module, making it accessible from the
layout or host. Typically used inside `csl.comptime` blocks.
Corresponds to `@export_symbol(sym, "name")` in CSL.

Example:
```mlir
csl.export_symbol @compute
csl.export_symbol @buf_ptr alias("data")
```

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>sym</code></td><td>::mlir::FlatSymbolRefAttr</td><td>flat symbol reference attribute</td></tr>
<tr><td><code>alias</code></td><td>::mlir::StringAttr</td><td>string attribute</td></tr>
</table>



### `csl.func` (xilinx::csl::FuncOp)

_CSL function definition_

Defines a CSL function on a PE. The body contains standard MLIR
operations (arith, scf, memref) representing the compute kernel.
Corresponds to `fn name() void { ... }` in CSL.

Example:
```mlir
csl.func @compute() {
  csl.return
}
```

Traits: `IsolatedFromAbove`

Interfaces: `Symbol`

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>sym_name</code></td><td>::mlir::StringAttr</td><td>string attribute</td></tr>
<tr><td><code>arg_attrs</code></td><td>::mlir::DictionaryAttr</td><td>dictionary of named attribute values</td></tr>
<tr><td><code>res_attrs</code></td><td>::mlir::DictionaryAttr</td><td>dictionary of named attribute values</td></tr>
</table>



### `csl.get_fab_dsd` (xilinx::csl::GetFabDsdOp)

_Create a fabric-backed Data Structure Descriptor_

Syntax:

```
operation ::= `csl.get_fab_dsd` $kind $color `,` $length `:` type($color) `,` type($length) `->` type($result)
              attr-dict
```

Creates a DSD backed by the on-chip fabric (network). Used for
streaming data to/from neighboring PEs via colors.
The `kind` attribute selects between fabric input (`fabin`) and
fabric output (`fabout`).

Example:
```mlir
%c = csl.color 0 : !csl.color
%dsd_in  = csl.get_fab_dsd fabin  %c, %len : !csl.color, index -> !csl.dsd
%dsd_out = csl.get_fab_dsd fabout %c, %len : !csl.color, index -> !csl.dsd
```

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>kind</code></td><td>xilinx::csl::DsdKindAttr</td><td>CSL DSD kind</td></tr>
</table>

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `color` | CSL color type |
| `length` | index |

#### Results:

| Result | Description |
| :----: | ----------- |
| `result` | CSL DSD type |



### `csl.get_mem_dsd` (xilinx::csl::GetMemDsdOp)

_Create a memory-backed Data Structure Descriptor_

Syntax:

```
operation ::= `csl.get_mem_dsd` $buffer `,` $length `:` type($buffer) `,` type($length) `->` type($result)
              attr-dict
```

Creates a DSD referencing a contiguous region of PE-local memory.
Corresponds to `@get_dsd(mem1d_dsd, ...)` in CSL.

Example:
```mlir
%dsd = csl.get_mem_dsd %buf, %len : memref<1024xf32>, index -> !csl.dsd
```

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `buffer` | memref of any type values |
| `length` | index |

#### Results:

| Result | Description |
| :----: | ----------- |
| `result` | CSL DSD type |



### `csl.hstack` (xilinx::csl::HstackOp)

_Stack code regions horizontally (left-to-right, touching)_

Syntax:

```
operation ::= `csl.hstack` `[` $code_regions `]` attr-dict
```

Corresponds to `layout.hstack([region_a, region_b, ...])`.
Regions are placed left-to-right with no gaps.

Example:
```mlir
csl.hstack [%r0, %r1, %r2]
```

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `code_regions` | variadic of CSL code region handle |



### `csl.import_module` (xilinx::csl::ImportModuleOp)

_Import a CSL library module_

Syntax:

```
operation ::= `csl.import_module` $module_name (`params` `(` $params^ `)`)? attr-dict `:` type($result)
```

Imports a CSL module by name, optionally passing compile-time
parameters. Corresponds to `@import_module("name", params)` in CSL.

Example:
```mlir
%mod = csl.import_module "<memcpy/memcpy>" : !csl.imported_module
%mod2 = csl.import_module "<memcpy/get_params>" params({width = 2 : i32}) : !csl.imported_module
```

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>module_name</code></td><td>::mlir::StringAttr</td><td>string attribute</td></tr>
<tr><td><code>params</code></td><td>::mlir::DictionaryAttr</td><td>dictionary of named attribute values</td></tr>
</table>

#### Results:

| Result | Description |
| :----: | ----------- |
| `result` | CSL imported module type |



### `csl.layout` (xilinx::csl::LayoutOp)

_Top-level CSL layout container_

Syntax:

```
operation ::= `csl.layout` $body attr-dict
```

Represents the CSL `layout { }` block that contains spatial configuration:
PE grid dimensions, tile-to-code assignments, and host-visible exports.

Example:
```mlir
csl.layout {
  csl.set_rectangle 2, 2
  csl.set_tile_code 0, 0 file("pe_program.csl")
  csl.export_name "data" : f32
}
```

Traits: `NoTerminator`, `SingleBlock`



### `csl.module` (xilinx::csl::ModuleOp)

_CSL module (PE program) definition_

Syntax:

```
operation ::= `csl.module` $sym_name $body attr-dict
```

Defines a CSL module that can be assigned to PE tiles. Contains
parameter declarations, variable declarations, functions, tasks,
and a comptime block.

Example:
```mlir
csl.module @pe_program {
  csl.param @memcpy_params : i64
  csl.var @buf : memref<1024xf32>
  csl.func @compute() { csl.return }
  csl.comptime { csl.export_symbol @compute }
}
```

Traits: `IsolatedFromAbove`, `NoTerminator`, `SingleBlock`, `SymbolTable`

Interfaces: `Symbol`

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>sym_name</code></td><td>::mlir::StringAttr</td><td>string attribute</td></tr>
</table>



### `csl.mov` (xilinx::csl::MovOp)

_Bulk DSD-to-DSD data move_

Syntax:

```
operation ::= `csl.mov` $dst `,` $src `:` type($dst) `,` type($src) attr-dict
```

Performs a hardware-accelerated bulk data move between two DSDs.
Corresponds to `@mov32` or `@mov16` builtins in CSL.

Example:
```mlir
csl.mov %dst_dsd, %src_dsd : !csl.dsd, !csl.dsd
```

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `dst` | CSL DSD type |
| `src` | CSL DSD type |



### `csl.paint` (xilinx::csl::PaintOp)

_Paint a single PE with a color and routing position(s)_

Syntax:

```
operation ::= `csl.paint` $region `at` `(` $pe_x `,` $pe_y `)` `color` `(` $color `)` `routing` `(` $routing_positions `)` attr-dict
```

Corresponds to `region.paint(IntVector(x, y), color, [rp, ...])`.

Example:
```mlir
%rp_s = csl.routing_position inputs([4]) outputs([2])
csl.paint %r at(0, 0) color(%c) routing(%rp_s)
```

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>pe_x</code></td><td>::mlir::IntegerAttr</td><td>64-bit signless integer attribute</td></tr>
<tr><td><code>pe_y</code></td><td>::mlir::IntegerAttr</td><td>64-bit signless integer attribute</td></tr>
</table>

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `region` | CSL code region handle |
| `color` | CSL color type |
| `routing_positions` | variadic of 32-bit signless integer |



### `csl.paint_all` (xilinx::csl::PaintAllOp)

_Paint every PE in a region with a color and routing positions_

Syntax:

```
operation ::= `csl.paint_all` $region `color` `(` $color `)` `routing` `(` $routing_positions `)` attr-dict
```

Corresponds to `region.paint_all(color, [rp, ...])`.

Example:
```mlir
%rp = csl.routing_position inputs([4]) outputs([2])
csl.paint_all %r color(%c) routing(%rp)
```

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `region` | CSL code region handle |
| `color` | CSL color type |
| `routing_positions` | variadic of 32-bit signless integer |



### `csl.paint_range` (xilinx::csl::PaintRangeOp)

_Paint a rectangular sub-range of PEs with a color_

Syntax:

```
operation ::= `csl.paint_range` $region `rect` `(` $rect_ox `,` $rect_oy `,` $rect_w `,` $rect_h `)`
              `color` `(` $color `)` `routing` `(` $routing_positions `)` attr-dict
```

Corresponds to `region.paint_range(IntRectangle(origin, dims), color, [rp, ...])`.
The rectangle is expressed as (origin_x, origin_y, width, height) in
region-local coordinates.

Example:
```mlir
%rp = csl.routing_position inputs([4]) outputs([2])
csl.paint_range %r rect(0, 0, 2, 1) color(%c) routing(%rp)
```

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>rect_ox</code></td><td>::mlir::IntegerAttr</td><td>64-bit signless integer attribute</td></tr>
<tr><td><code>rect_oy</code></td><td>::mlir::IntegerAttr</td><td>64-bit signless integer attribute</td></tr>
<tr><td><code>rect_w</code></td><td>::mlir::IntegerAttr</td><td>64-bit signless integer attribute</td></tr>
<tr><td><code>rect_h</code></td><td>::mlir::IntegerAttr</td><td>64-bit signless integer attribute</td></tr>
</table>

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `region` | CSL code region handle |
| `color` | CSL color type |
| `routing_positions` | variadic of 32-bit signless integer |



### `csl.param` (xilinx::csl::ParamOp)

_Declare a compile-time module parameter_

Syntax:

```
operation ::= `csl.param` $sym_name `:` $type attr-dict
```

Declares a compile-time parameter for a CSL module. Parameters
are specialized per-tile when assigned via `csl.set_tile_code`.
Corresponds to `param name: type;` in CSL.

Example:
```mlir
csl.param @memcpy_params : i64
csl.param @tile_id : i32
```

Interfaces: `Symbol`

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>sym_name</code></td><td>::mlir::StringAttr</td><td>string attribute</td></tr>
<tr><td><code>type</code></td><td>::mlir::TypeAttr</td><td>any type attribute</td></tr>
</table>



### `csl.place` (xilinx::csl::PlaceOp)

_Place a code region at a global grid coordinate_

Syntax:

```
operation ::= `csl.place` $region `at` `(` $x `,` $y `)` attr-dict
```

Corresponds to `region.place(x, y)`. Fixes the top-left corner of the
code region at global PE coordinate (x, y).

Example:
```mlir
csl.place %r at(2, 4)
```

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>x</code></td><td>::mlir::IntegerAttr</td><td>64-bit signless integer attribute</td></tr>
<tr><td><code>y</code></td><td>::mlir::IntegerAttr</td><td>64-bit signless integer attribute</td></tr>
</table>

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `region` | CSL code region handle |



### `csl.return` (xilinx::csl::ReturnOp)

_Return from a CSL function or task_

Syntax:

```
operation ::= `csl.return` attr-dict
```

Traits: `HasParent<FuncOp, TaskOp>`, `Terminator`



### `csl.route` (xilinx::csl::RouteOp)

_Configure routing direction for a color_

Syntax:

```
operation ::= `csl.route` $color `dir` `(` $direction `)` attr-dict
```

Binds a color to a fabric direction (NORTH, SOUTH, EAST, WEST, RAMP).
This determines how wavelets tagged with this color propagate through
the on-chip network.

Example:
```mlir
%c = csl.color 0 : !csl.color
csl.route %c dir(EAST)
```

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>direction</code></td><td>xilinx::csl::DirectionAttr</td><td>CSL routing direction</td></tr>
</table>

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `color` | CSL color type |



### `csl.routing_position` (xilinx::csl::RoutingPositionOp)

_Build a RoutingPosition (per-PE input/output direction set)_

Syntax:

```
operation ::= `csl.routing_position` `inputs` `(` $input_dirs `)` `outputs` `(` $output_dirs `)` attr-dict `:` type($result)
```

Models `RoutingPosition().set_input([...]).set_output([...])`.
Direction values follow the Direction enum: NORTH=0, SOUTH=1, EAST=2,
WEST=3, RAMP=4 (RAMP = PE compute unit).

The result is an i32 SSA value used as an opaque handle by paint ops.

Examples:
```mlir
// sender: data from PE compute out to EAST
%rp_send = csl.routing_position inputs([4]) outputs([2])
// receiver: data from WEST into PE compute
%rp_recv = csl.routing_position inputs([3]) outputs([4])
// output port: data from PE compute (no explicit output direction)
%rp_out  = csl.routing_position inputs([4]) outputs([])
// input port: data into PE compute (no explicit input direction)
%rp_in   = csl.routing_position inputs([]) outputs([4])
```

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>input_dirs</code></td><td>::mlir::ArrayAttr</td><td>32-bit integer array attribute</td></tr>
<tr><td><code>output_dirs</code></td><td>::mlir::ArrayAttr</td><td>32-bit integer array attribute</td></tr>
</table>

#### Results:

| Result | Description |
| :----: | ----------- |
| `result` | 32-bit signless integer |



### `csl.scoped_color` (xilinx::csl::ScopedColorOp)

_Declare a region-scoped symbolic color_

Syntax:

```
operation ::= `csl.scoped_color` $region $sym_name attr-dict `:` type($result)
```

Corresponds to `region.color('name')`. The resulting color is globally
unique (namespaced by region), preventing physical ID collisions when
multiple regions each declare a color with the same local name.

Example:
```mlir
%c = csl.scoped_color %r "tx" : !csl.color
```

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>sym_name</code></td><td>::mlir::StringAttr</td><td>string attribute</td></tr>
</table>

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `region` | CSL code region handle |

#### Results:

| Result | Description |
| :----: | ----------- |
| `result` | CSL color type |



### `csl.sdk_layout` (xilinx::csl::SdkLayoutOp)

_Top-level SdkLayout container (Python API model)_

Syntax:

```
operation ::= `csl.sdk_layout` $body attr-dict
```

Models the Python `SdkLayout` object. The body holds code_region
creation, placement, routing, connections, and stream declarations.

Example:
```mlir
csl.sdk_layout {
  %r = csl.code_region "./pe.csl" name("pe") width(2) height(1)
  %c = csl.sym_color "tx" : !csl.color
  %rp = csl.routing_position inputs([4]) outputs([2])
  %port = csl.create_output_port %r color(%c) edge(RIGHT) routing(%rp) size(10)
}
```

Traits: `NoTerminator`, `SingleBlock`



### `csl.set_param` (xilinx::csl::SetParamOp)

_Set a compile-time parameter on a single PE_

Syntax:

```
operation ::= `csl.set_param` $region `at` `(` $pe_x `,` $pe_y `)` `param` `(` $param_name `)` `value` `(` $value `)` attr-dict
```

Corresponds to `region.set_param(IntVector(x, y), 'name', value)`.

Example:
```mlir
csl.set_param %r at(0, 0) param("select") value(0 : i64)
csl.set_param %r at(1, 0) param("select") value(1 : i64)
```

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>pe_x</code></td><td>::mlir::IntegerAttr</td><td>64-bit signless integer attribute</td></tr>
<tr><td><code>pe_y</code></td><td>::mlir::IntegerAttr</td><td>64-bit signless integer attribute</td></tr>
<tr><td><code>param_name</code></td><td>::mlir::StringAttr</td><td>string attribute</td></tr>
<tr><td><code>value</code></td><td>::mlir::IntegerAttr</td><td>64-bit signless integer attribute</td></tr>
</table>

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `region` | CSL code region handle |



### `csl.set_param_all` (xilinx::csl::SetParamAllOp)

_Set a compile-time parameter on every PE in a region_

Syntax:

```
operation ::= `csl.set_param_all` $region (`param` `(` $param_name^ `)` `value` `(` $value `)`)? (`color` `(` $color^ `)`)? attr-dict
```

Corresponds to `region.set_param_all('name', value)` for integer params
or `region.set_param_all(color_obj)` to inject a resolved color ID.

Exactly one of (param_name+value) or (color) must be provided.

Examples:
```mlir
csl.set_param_all %r param("size") value(10 : i64)
csl.set_param_all %r color(%c)
```

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>param_name</code></td><td>::mlir::StringAttr</td><td>string attribute</td></tr>
<tr><td><code>value</code></td><td>::mlir::IntegerAttr</td><td>64-bit signless integer attribute</td></tr>
</table>

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `region` | CSL code region handle |
| `color` | CSL color type |



### `csl.set_rectangle` (xilinx::csl::SetRectangleOp)

_Define PE grid dimensions_

Syntax:

```
operation ::= `csl.set_rectangle` $width `,` $height attr-dict
```

Sets the rectangular PE grid dimensions for the CSL program.
Corresponds to `@set_rectangle(width, height)` in CSL.

Example:
```mlir
csl.set_rectangle 4, 8
```

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>width</code></td><td>::mlir::IntegerAttr</td><td>64-bit signless integer attribute</td></tr>
<tr><td><code>height</code></td><td>::mlir::IntegerAttr</td><td>64-bit signless integer attribute</td></tr>
</table>



### `csl.set_tile_code` (xilinx::csl::SetTileCodeOp)

_Assign a CSL module to a PE at grid position (x, y)_

Syntax:

```
operation ::= `csl.set_tile_code` $x `,` $y `file` `(` $file `)` (`params` `(` $params^ `)`)? attr-dict
```

Maps a CSL source file to a specific PE tile in the grid. Corresponds
to `@set_tile_code(x, y, "file.csl", .{ params })` in CSL.

The optional `params` dictionary carries compile-time parameters
forwarded to the PE module.

Example:
```mlir
csl.set_tile_code 0, 1 file("pe_program.csl")
csl.set_tile_code 1, 1 file("pe_program.csl") params({col = 1 : i32})
```

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>x</code></td><td>::mlir::IntegerAttr</td><td>64-bit signless integer attribute</td></tr>
<tr><td><code>y</code></td><td>::mlir::IntegerAttr</td><td>64-bit signless integer attribute</td></tr>
<tr><td><code>file</code></td><td>::mlir::StringAttr</td><td>string attribute</td></tr>
<tr><td><code>params</code></td><td>::mlir::DictionaryAttr</td><td>dictionary of named attribute values</td></tr>
</table>



### `csl.sym_color` (xilinx::csl::SymColorOp)

_Declare a free-standing symbolic color_

Syntax:

```
operation ::= `csl.sym_color` $sym_name attr-dict `:` type($result)
```

Corresponds to `Color('name')`. The compiler assigns a physical color ID
at compile time. The resulting !csl.color can be passed to paint, port,
and set_param_all ops.

Example:
```mlir
%c = csl.sym_color "tx" : !csl.color
```

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>sym_name</code></td><td>::mlir::StringAttr</td><td>string attribute</td></tr>
</table>

#### Results:

| Result | Description |
| :----: | ----------- |
| `result` | CSL color type |



### `csl.task` (xilinx::csl::TaskOp)

_CSL event-driven task bound to a color_

Defines an event-driven task on a PE. Tasks are triggered when a
wavelet arrives on the bound color. This is the fundamental
execution mechanism on the WSE.

Example:
```mlir
csl.task @recv_task() color(3) {
  csl.return
}
```

Traits: `IsolatedFromAbove`

Interfaces: `Symbol`

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>sym_name</code></td><td>::mlir::StringAttr</td><td>string attribute</td></tr>
<tr><td><code>color_id</code></td><td>::mlir::IntegerAttr</td><td>32-bit signless integer attribute</td></tr>
</table>



### `csl.var` (xilinx::csl::VarOp)

_Declare a PE-local variable_

Syntax:

```
operation ::= `csl.var` $sym_name `:` $type attr-dict
```

Declares a variable in the PE's local memory. Corresponds to
`var name: [N]T` in CSL.

Example:
```mlir
csl.var @buf : memref<1024xf32>
csl.var @matrix : memref<32x32xf16>
```

Interfaces: `Symbol`

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>sym_name</code></td><td>::mlir::StringAttr</td><td>string attribute</td></tr>
<tr><td><code>type</code></td><td>::mlir::TypeAttr</td><td>any type attribute</td></tr>
</table>



### `csl.vstack` (xilinx::csl::VstackOp)

_Stack code regions vertically (top-to-bottom, touching)_

Syntax:

```
operation ::= `csl.vstack` `[` $code_regions `]` (`origin` `(` $origin_x^ `,` $origin_y `)`)? attr-dict
```

Corresponds to `layout.vstack([region_a, region_b, ...], origin=IntVector(ox, oy))`.
The optional `origin` attribute fixes the top-left corner of the stacked group.

Examples:
```mlir
csl.vstack [%r0, %r1] origin(0, 5)
csl.vstack [%r0, %r1]
```

#### Attributes:

<table>
<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>
<tr><td><code>origin_x</code></td><td>::mlir::IntegerAttr</td><td>64-bit signless integer attribute</td></tr>
<tr><td><code>origin_y</code></td><td>::mlir::IntegerAttr</td><td>64-bit signless integer attribute</td></tr>
</table>

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `code_regions` | variadic of CSL code region handle |



## Type constraints

### CSL code region handle


### CSL color type


### CSL DSD type


### CSL imported module type


### CSL port handle


### CSL stream handle


## Enums

### Direction

_CSL routing direction_

#### Cases:

| Symbol | Value | String |
| :----: | :---: | ------ |
| NORTH | `0` | NORTH |
| SOUTH | `1` | SOUTH |
| EAST | `2` | EAST |
| WEST | `3` | WEST |
| RAMP | `4` | RAMP |

### DsdKind

_CSL DSD kind_

#### Cases:

| Symbol | Value | String |
| :----: | :---: | ------ |
| mem1d | `0` | mem1d |
| mem2d | `1` | mem2d |
| fabin | `2` | fabin |
| fabout | `3` | fabout |

### Edge

_CSL port edge (SdkLayout)_

#### Cases:

| Symbol | Value | String |
| :----: | :---: | ------ |
| LEFT | `0` | LEFT |
| RIGHT | `1` | RIGHT |
| TOP | `2` | TOP |
| BOTTOM | `3` | BOTTOM |


---



---

## 4. Design Evaluation and Roadmap

# CSL Dialect Design Evaluation & Roadmap

Based on the recent layout ops stress testing and a review of the Cerebras CSL Language Guide, several design flaws (missing verifiers) and missing language features have been identified. 

## 1. Identified Flaws in Current Layout Ops

When pushing the MLIR ops to edge cases (see `mlir/test/Dialect/CSL/layout_stress.mlir`), the parser/verifier currently accepts invalid spatial configurations. We need to add MLIR C++ verifiers for the following:

*   **Out-of-Bounds Painting**: `csl.paint pe(x, y)` inside a `csl.code_region shape(W, H)` allows `x >= W` or `y >= H`. MLIR should reject this at compile time.
*   **Placement Overlaps**: `csl.place` allows multiple regions to be placed on overlapping absolute WSE coordinates. We need a global layout verifier to check bounding box collisions.
*   **Dataflow Cycle/Type Checking**: `csl.dataflow %src -> %dst` should verify that `%src` is an "output" port, `%dst` is an "input" port, and potentially that their sizes match.

## 2. Missing CSL Language Features

To fully generate the CSL `language_index` and address your comments regarding "TILE SELECTION" and "SCHEDULE/ALGORITHM", our MLIR Dialect needs the following new features:

### A. Advanced Types & Arrays (Bundles)
*   **Arrays of Colors/Routes**: Currently, we pass variadic single colors. CSL heavily relies on arrays of colors for wide channels (e.g., `var colors [3]color`). We need an array/bundle abstraction.
*   **Structs/Enums**: `!csl.struct` and `!csl.enum` types are needed for passing complex configuration payloads to kernels.

### B. Tile Specialization (Compute vs Memory)
*   You mentioned: *"Say make this tile a memory tile only, Say make this a compute only"*. 
*   **Solution**: We can add a `kind` attribute to `csl.code_region` (e.g., `kind = "compute"`, `kind = "memory"`) or introduce specific container ops like `csl.mem_region` vs `csl.compute_region` to enforce what inner operations are valid (e.g., `csl.task` only allowed in compute).

### C. State Machines & Micro-Thread Control Flow
*   You mentioned: *"Wait until task1 is finished, Now perform task2... This can be a state machine maybe? in a loop"*
*   **Solution**: CSL has native `struct fsm` constructs and block/unblock semantics. We need:
    *   `csl.state_machine` / `csl.state` ops.
    *   `csl.block` / `csl.unblock` for task synchronization.
    *   `csl.loop` (or integration with standard `scf.for`) that correctly translates to CSL hardware loop constructs.

## 3. Recommended Next Steps

1.  **Phase 1**: Add C++ verifiers (`OpTrait` or custom `verify()` methods) to `CSLLayoutOps.cpp` to fix the layout flaws (OOB painting, collisions).
2.  **Phase 2**: Introduce Tile Specialization attributes to `csl.code_region`.
3.  **Phase 3**: Introduce State Machine and Control Flow ops to `CSLKernelOps.td`.
