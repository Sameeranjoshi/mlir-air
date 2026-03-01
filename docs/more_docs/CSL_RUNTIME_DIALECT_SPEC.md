# CSL Runtime (csl_rt) Dialect Specification

## Overview

The **CSL Runtime dialect** (`csl_rt`) provides a structured MLIR representation of operations that directly correspond to the Cerebras SDK's **SdkLayout** and **SdkRuntime** Python APIs. It separates generic spatial/semantic layout operations (in the **CSL dialect**) from host-facing runtime operations (layout compilation, runtime creation, data movement, kernel launch).

## Dialect Name and Namespace

- **MLIR Dialect Name:** `csl_rt`
- **C++ Namespace:** `xilinx::csl_rt`

## Types

All types are opaque handles representing Python SDK objects.

| Type | MLIR Syntax | Corresponds To | Purpose |
|------|-------------|---|---|
| `LayoutType` | `!csl_rt.layout` | `SdkLayout` instance | Layout object for spatial placement |
| `CodeRegionType` | `!csl_rt.code_region` | `CodeRegion` object | Code region with placement and parameters |
| `CompileArtifactsType` | `!csl_rt.compile_artifacts` | `SdkCompileArtifacts` | Compiled layout artifacts |
| `RuntimeType` | `!csl_rt.runtime` | `SdkRuntime` instance | Runtime for kernel execution |
| `ColorType` | `!csl_rt.color` | `Color` | Color handle (for routing, optional) |
| `RoutingPositionType` | `!csl_rt.routing_position` | `RoutingPosition` | Routing configuration (optional) |
| `PortType` | `!csl_rt.port` | `PortHandle` | Port handle (optional) |
| `StreamType` | `!csl_rt.stream` | Stream handle | Stream handle (optional) |

## Operations

### Layout Operations (SdkLayout API)

Map 1:1 to methods on `SdkLayout` and `CodeRegion` objects.

#### `csl_rt.create_layout`
Creates a layout instance.

**Syntax:** `%layout = csl_rt.create_layout : () -> !csl_rt.layout`

**Maps to:** `SdkLayout()` or `SdkLayout(platform)`

**Arguments:** None (optional platform attribute in future)

**Results:**
- `%layout`: Layout instance

---

#### `csl_rt.create_code_region`
Creates a code region on a layout.

**Syntax:**
```mlir
%code_region = csl_rt.create_code_region %layout "source.csl", "region_name", 16 : index, 16 : index : (!csl_rt.layout) -> !csl_rt.code_region
```

**Maps to:** `layout.create_code_region(source, name, width, height)`

**Arguments:**
- `%layout`: Layout instance
- `"source.csl"`: String attribute (CSL source file path)
- `"region_name"`: String attribute (region name)
- `width`: Index attribute (width in PEs)
- `height`: Index attribute (height in PEs)

**Results:**
- `%code_region`: Code region instance

---

#### `csl_rt.place`
Places a code region at a location.

**Syntax:**
```mlir
%placed = csl_rt.place %code_region at (0, 0) : (!csl_rt.code_region) -> !csl_rt.code_region
```

**Maps to:** `code_region.place(x, y)`

**Arguments:**
- `%code_region`: Code region instance
- `x`: Index attribute (x coordinate)
- `y`: Index attribute (y coordinate)

**Results:**
- `%placed`: Placed code region (same type, for chaining)

---

#### `csl_rt.set_param_all`
Sets a parameter for a code region.

**Syntax:**
```mlir
%params = csl_rt.set_param_all %region "width" = 16 : (!csl_rt.code_region) -> !csl_rt.code_region
```

**Maps to:** `code_region.set_param_all(name, value)`

**Arguments:**
- `%region`: Code region instance
- `"width"`: String attribute (parameter name)
- `16`: Attribute (parameter value, int/float)

**Results:**
- `%params`: Updated code region

---

#### `csl_rt.export_name`
Exports a symbol from a layout.

**Syntax:**
```mlir
%exported = csl_rt.export_name %layout "symbol", "f32" : (!csl_rt.layout) -> !csl_rt.layout
```

**Maps to:** `layout.export_name("symbol", "type_spec")`

**Arguments:**
- `%layout`: Layout instance
- `"symbol"`: String attribute (symbol name)
- `"f32"`: String attribute (type specification)

**Results:**
- `%exported`: Updated layout

---

#### `csl_rt.compile`
Compiles a layout.

**Syntax:**
```mlir
%artifacts = csl_rt.compile %layout : (!csl_rt.layout) -> !csl_rt.compile_artifacts
```

**Maps to:** `layout.compile(out_prefix='out')`

**Arguments:**
- `%layout`: Layout instance
- Optional `out_prefix`: String attribute (output directory prefix)

**Results:**
- `%artifacts`: Compile artifacts

---

### Runtime Operations (SdkRuntime API)

Map 1:1 to methods on `SdkRuntime` instance.

#### `csl_rt.runtime_create`
Creates a runtime instance.

**Syntax:**
```mlir
%runtime = csl_rt.runtime_create %artifacts : (!csl_rt.compile_artifacts) -> !csl_rt.runtime
```

**Maps to:** `SdkRuntime(artifacts, platform, ...)`

**Arguments:**
- `%artifacts`: Compile artifacts

**Results:**
- `%runtime`: Runtime instance

---

#### `csl_rt.load`
Loads the runtime (initializes).

**Syntax:**
```mlir
%loaded = csl_rt.load %runtime : (!csl_rt.runtime) -> !csl_rt.runtime
```

**Maps to:** `runtime.load()`

**Arguments:**
- `%runtime`: Runtime instance

**Results:**
- `%loaded`: Runtime instance (for chaining)

---

#### `csl_rt.run`
Runs the runtime.

**Syntax:**
```mlir
%ran = csl_rt.run %runtime : (!csl_rt.runtime) -> !csl_rt.runtime
```

**Maps to:** `runtime.run()`

**Arguments:**
- `%runtime`: Runtime instance

**Results:**
- `%ran`: Runtime instance

---

#### `csl_rt.stop`
Stops the runtime.

**Syntax:**
```mlir
csl_rt.stop %runtime : !csl_rt.runtime
```

**Maps to:** `runtime.stop()`

**Arguments:**
- `%runtime`: Runtime instance

**Results:** None (terminator-like op)

---

#### `csl_rt.get_id`
Gets a symbol ID from the runtime.

**Syntax:**
```mlir
%id = csl_rt.get_id %runtime "symbol" : (!csl_rt.runtime) -> i32
```

**Maps to:** `runtime.get_id("symbol")`

**Arguments:**
- `%runtime`: Runtime instance
- `"symbol"`: String attribute (symbol name)

**Results:**
- `%id`: i32 (symbol ID)

---

#### `csl_rt.memcpy_h2d`
Host-to-device memory copy.

**Syntax:**
```mlir
%after = csl_rt.memcpy_h2d %runtime 0 "src" at (0, 0) with_size (16, 16) elem_per_pe 256 : (!csl_rt.runtime) -> !csl_rt.runtime
```

**Maps to:** `runtime.memcpy_h2d(dest_id, src, px, py, w, h, elem_per_pe, ...)`

**Arguments:**
- `%runtime`: Runtime instance
- `dest_id`: i32 attribute (destination symbol ID)
- `"src"`: String attribute (source array name, template placeholder)
- `px`, `py`: Index attributes (region origin)
- `w`, `h`: Index attributes (region size)
- `elem_per_pe`: Index attribute (elements per PE)

**Results:**
- `%after`: Runtime instance

---

#### `csl_rt.memcpy_d2h`
Device-to-host memory copy.

**Syntax:**
```mlir
%after = csl_rt.memcpy_d2h %runtime "dest" from (0, 0) with_size (16, 16) elem_per_pe 256 : (!csl_rt.runtime) -> !csl_rt.runtime
```

**Maps to:** `runtime.memcpy_d2h(dest, src_id, px, py, w, h, elem_per_pe, ...)`

**Arguments:**
- `%runtime`: Runtime instance
- `"dest"`: String attribute (destination array name, template placeholder)
- `src_id`: i32 attribute (source symbol ID)
- `px`, `py`: Index attributes (region origin)
- `w`, `h`: Index attributes (region size)
- `elem_per_pe`: Index attribute (elements per PE)

**Results:**
- `%after`: Runtime instance

---

#### `csl_rt.launch`
Launches a host-callable function.

**Syntax:**
```mlir
%after = csl_rt.launch %runtime "compute" : (!csl_rt.runtime) -> !csl_rt.runtime
```

**Maps to:** `runtime.launch("symbol", nonblock=False)`

**Arguments:**
- `%runtime`: Runtime instance
- `"compute"`: String attribute (function/kernel name)
- Optional `nonblock`: Bool attribute (blocking/non-blocking)

**Results:**
- `%after`: Runtime instance

---

## Canonical MLIR Form

All operations use **declarative `assemblyFormat`** (no custom parsers/printers). Standard pattern:

```mlir
%result = csl_rt.op %arg0, %arg1 "attr_value" at (x, y) with_size (w, h) elem_per_pe N : (type0, type1) -> result_type
```

Example chaining (GEMV-05):
```mlir
%layout = csl_rt.create_layout : () -> !csl_rt.layout
%region = csl_rt.create_code_region %layout "pe.csl", "gemv", 16 : index, 16 : index : (!csl_rt.layout) -> !csl_rt.code_region
%placed = csl_rt.place %region at (0, 0) : (!csl_rt.code_region) -> !csl_rt.code_region
%artifacts = csl_rt.compile %layout : (!csl_rt.layout) -> !csl_rt.compile_artifacts
%runtime = csl_rt.runtime_create %artifacts : (!csl_rt.compile_artifacts) -> !csl_rt.runtime
%loaded = csl_rt.load %runtime : (!csl_rt.runtime) -> !csl_rt.runtime
%after_h2d = csl_rt.memcpy_h2d %loaded 0 "A" at (0, 0) with_size (16, 16) elem_per_pe 256 : (!csl_rt.runtime) -> !csl_rt.runtime
%after_launch = csl_rt.launch %after_h2d "compute" : (!csl_rt.runtime) -> !csl_rt.runtime
%after_d2h = csl_rt.memcpy_d2h %after_launch "y" from (0, 0) with_size (1, 16) elem_per_pe 16 : (!csl_rt.runtime) -> !csl_rt.runtime
csl_rt.stop %after_d2h : !csl_rt.runtime
```

## Python Emission

Operations are translated to Python via `CSLRuntimeToPy` backend:

### Layout Emission (layout.py)
```python
from cerebras.sdk.sdk_runtime import SdkLayout

layout = SdkLayout()
region = layout.create_code_region('pe.csl', 'gemv', 16, 16)
region.place(0, 0)
region.set_param_all('width', 16)
region.set_param_all('M', 256)
layout.export_name('A', 'f32')
artifacts = layout.compile(out_prefix='out')
```

### Runtime Emission (run.py template)
```python
from cerebras.sdk.sdk_runtime import SdkRuntime

artifacts = ...  # from layout.compile()
runtime = SdkRuntime(artifacts, platform='wse3')
runtime.load()

# Template placeholders:
# A_data = ...  # User fills with actual array
# runtime.memcpy_h2d(0, A_data, 0, 0, 16, 16, 256)

id_A = runtime.get_id('A')
runtime.launch('compute')

# y_result = ...  # User receives result
# runtime.memcpy_d2h(y_result, id_A, 0, 0, 1, 16, 16)

runtime.stop()
```

## Example: GEMV-05 (Multiple PEs)

See `mlir/test/Dialect/CSLRuntime/gemv05_example.mlir` for a complete end-to-end example that:
- Creates a 16×16 PE layout
- Sets kernel parameters (width, M, N)
- Exports symbols for host I/O (A, x, b, y, compute)
- Compiles the layout
- Creates runtime and loads
- Performs H2D memcpy for inputs
- Launches compute kernel
- Performs D2H memcpy for results
- Stops runtime

## Conversion from CSL Dialect

The **CSL → csl_rt lowering pass** (`csl-to-csl-rt`) converts semantic CSL layout operations to runtime operations:

- `csl.spatial_placement` → sequence of `csl_rt.create_layout`, `csl_rt.create_code_region`, `csl_rt.place`, `csl_rt.set_param_all`, `csl_rt.export_name`, `csl_rt.compile`
- Supports minimal path (one code region per layout)
- TODO: Extend to multiple regions, ports, streams

## Future Extensions

- Color allocation and routing operations
- Port and stream creation
- Data movement templates (bulk transfers, streaming)
- Error handling and validation
