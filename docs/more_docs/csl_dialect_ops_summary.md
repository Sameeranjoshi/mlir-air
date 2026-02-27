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
