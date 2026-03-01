//===- CSLRuntimeOps.td - CSL Runtime operations ----------*- tablegen -*-===//
//
// Part of the air-to-csl project.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
//
// Layout and runtime operations for the CSL Runtime dialect.
// All ops use declarative assemblyFormat only (no custom parse/print).
//
//===----------------------------------------------------------------------===//

#ifndef CSLRUNTIME_OPS
#define CSLRUNTIME_OPS

include "air/Dialect/CSLRuntime/CSLRuntimeBase.td"
include "mlir/IR/BuiltinAttributes.td"
include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// Layout Operations (SdkLayout API)
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// csl_rt.create_layout
//===----------------------------------------------------------------------===//
// Creates a layout instance. Maps to SdkLayout(platform) or SdkLayout().
// Result: !csl_rt.layout
def CSLRuntime_CreateLayoutOp : CSLRuntime_Op<"create_layout"> {
  let summary = "Create a layout instance";
  let description = [{
    Creates a layout instance corresponding to SdkLayout(platform) if a
    platform operand is provided, or SdkLayout() for standalone layout.
  }];

  let results = (outs CSLRuntime_LayoutType:$layout);
  let assemblyFormat = "attr-dict `:` type($layout)";
}

//===----------------------------------------------------------------------===//
// csl_rt.create_code_region
//===----------------------------------------------------------------------===//
// Creates a code region on a layout.
// Maps to layout.create_code_region(source, name, width, height).
def CSLRuntime_CreateCodeRegionOp : CSLRuntime_Op<"create_code_region"> {
  let summary = "Create a code region on a layout";
  let description = [{
    Creates a code region on a layout with a given source file, name, and
    rectangular dimensions. Maps to layout.create_code_region(source, name,
    width, height).
  }];

  let arguments = (ins CSLRuntime_LayoutType:$layout,
                       StrAttr:$source,
                       StrAttr:$name,
                       IndexAttr:$width,
                       IndexAttr:$height);
  let results = (outs CSLRuntime_CodeRegionType:$code_region);
  let assemblyFormat = "$layout $source `,` $name `,` $width `,` $height "
                       "attr-dict `:` type($layout) `->` type($code_region)";
}

//===----------------------------------------------------------------------===//
// csl_rt.place
//===----------------------------------------------------------------------===//
// Places a code region at a (x, y) location.
// Maps to code_region.place(x, y).
def CSLRuntime_PlaceOp : CSLRuntime_Op<"place"> {
  let summary = "Place a code region at a location";
  let description = [{
    Places a code region at a rectangular region (x, y) with given width and
    height. Maps to code_region.place(x, y) in SdkLayout.
  }];

  let arguments = (ins CSLRuntime_CodeRegionType:$code_region,
                       IndexAttr:$x,
                       IndexAttr:$y);
  let results = (outs CSLRuntime_CodeRegionType:$result);
  let assemblyFormat = "$code_region `at` `(` $x `,` $y `)` "
                       "attr-dict `:` type($code_region) `->` type($result)";
}

//===----------------------------------------------------------------------===//
// csl_rt.set_param_all
//===----------------------------------------------------------------------===//
// Sets a parameter for a code region.
// Maps to code_region.set_param_all(name, value).
def CSLRuntime_SetParamAllOp : CSLRuntime_Op<"set_param_all"> {
  let summary = "Set a parameter for a code region";
  let description = [{
    Sets a parameter (by name and value) for a code region. Maps to
    code_region.set_param_all(name, value) in SdkLayout.
  }];

  let arguments = (ins CSLRuntime_CodeRegionType:$code_region,
                       StrAttr:$param_name,
                       AnyAttr:$param_value);
  let results = (outs CSLRuntime_CodeRegionType:$result);
  let assemblyFormat = "$code_region $param_name `=` $param_value "
                       "attr-dict `:` type($code_region) `->` type($result)";
}

//===----------------------------------------------------------------------===//
// csl_rt.export_name
//===----------------------------------------------------------------------===//
// Exports a symbol from a layout.
// Maps to layout.export_name("name", "type_spec").
def CSLRuntime_ExportNameOp : CSLRuntime_Op<"export_name"> {
  let summary = "Export a symbol from a layout";
  let description = [{
    Exports a symbol (by name and type spec) from a layout. Maps to
    layout.export_name("name", "type_spec") in SdkLayout.
  }];

  let arguments = (ins CSLRuntime_LayoutType:$layout,
                       StrAttr:$symbol_name,
                       StrAttr:$symbol_type);
  let results = (outs CSLRuntime_LayoutType:$result);
  let assemblyFormat = "$layout $symbol_name `,` $symbol_type "
                       "attr-dict `:` type($layout) `->` type($result)";
}

//===----------------------------------------------------------------------===//
// csl_rt.compile
//===----------------------------------------------------------------------===//
// Compiles a layout.
// Maps to layout.compile(out_prefix='out').
def CSLRuntime_CompileOp : CSLRuntime_Op<"compile"> {
  let summary = "Compile a layout";
  let description = [{
    Compiles a layout and returns compile artifacts. Maps to
    layout.compile(out_prefix='out') in SdkLayout.
  }];

  let arguments = (ins CSLRuntime_LayoutType:$layout,
                       OptionalAttr<StrAttr>:$out_prefix);
  let results = (outs CSLRuntime_CompileArtifactsType:$artifacts);
  let assemblyFormat = "$layout attr-dict `:` type($layout) `->` type($artifacts)";
}

//===----------------------------------------------------------------------===//
// Runtime Operations (SdkRuntime API)
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// csl_rt.runtime_create
//===----------------------------------------------------------------------===//
// Creates a runtime instance.
// Maps to SdkRuntime(compile_artifacts, platform, ...).
def CSLRuntime_RuntimeCreateOp : CSLRuntime_Op<"runtime_create"> {
  let summary = "Create a runtime instance";
  let description = [{
    Creates a runtime instance from compile artifacts and optionally a platform.
    Maps to SdkRuntime(compile_artifacts, platform, ...) in SdkRuntime API.
  }];

  let arguments = (ins CSLRuntime_CompileArtifactsType:$artifacts);
  let results = (outs CSLRuntime_RuntimeType:$runtime);
  let assemblyFormat = "$artifacts attr-dict `:` type($artifacts) `->` type($runtime)";
}

//===----------------------------------------------------------------------===//
// csl_rt.load
//===----------------------------------------------------------------------===//
// Loads the runtime.
// Maps to runtime.load().
def CSLRuntime_LoadOp : CSLRuntime_Op<"load"> {
  let summary = "Load the runtime";
  let description = [{
    Loads the runtime. Maps to runtime.load() in SdkRuntime API.
  }];

  let arguments = (ins CSLRuntime_RuntimeType:$runtime);
  let results = (outs CSLRuntime_RuntimeType:$result);
  let assemblyFormat = "$runtime attr-dict `:` type($runtime) `->` type($result)";
}

//===----------------------------------------------------------------------===//
// csl_rt.run
//===----------------------------------------------------------------------===//
// Runs the runtime.
// Maps to runtime.run().
def CSLRuntime_RunOp : CSLRuntime_Op<"run"> {
  let summary = "Run the runtime";
  let description = [{
    Runs the runtime. Maps to runtime.run() in SdkRuntime API.
  }];

  let arguments = (ins CSLRuntime_RuntimeType:$runtime);
  let results = (outs CSLRuntime_RuntimeType:$result);
  let assemblyFormat = "$runtime attr-dict `:` type($runtime) `->` type($result)";
}

//===----------------------------------------------------------------------===//
// csl_rt.stop
//===----------------------------------------------------------------------===//
// Stops the runtime.
// Maps to runtime.stop().
def CSLRuntime_StopOp : CSLRuntime_Op<"stop"> {
  let summary = "Stop the runtime";
  let description = [{
    Stops the runtime. Maps to runtime.stop() in SdkRuntime API.
  }];

  let arguments = (ins CSLRuntime_RuntimeType:$runtime);
  let assemblyFormat = "$runtime attr-dict `:` type($runtime)";
}

//===----------------------------------------------------------------------===//
// csl_rt.get_id
//===----------------------------------------------------------------------===//
// Gets a symbol ID from the runtime.
// Maps to runtime.get_id("symbol_name").
def CSLRuntime_GetIdOp : CSLRuntime_Op<"get_id"> {
  let summary = "Get a symbol ID from the runtime";
  let description = [{
    Gets a symbol ID (integer) for a named symbol. Maps to
    runtime.get_id("symbol_name") in SdkRuntime API.
  }];

  let arguments = (ins CSLRuntime_RuntimeType:$runtime,
                       StrAttr:$symbol_name);
  let results = (outs I32:$id);
  let assemblyFormat = "$runtime $symbol_name attr-dict `:` type($runtime) `->` type($id)";
}

//===----------------------------------------------------------------------===//
// csl_rt.memcpy_h2d
//===----------------------------------------------------------------------===//
// Host-to-device memcpy.
// Maps to runtime.memcpy_h2d(dest, src, px, py, w, h, elem_per_pe, ...).
def CSLRuntime_MemcpyH2dOp : CSLRuntime_Op<"memcpy_h2d"> {
  let summary = "Host-to-device memcpy";
  let description = [{
    Performs a host-to-device memcpy. Maps to
    runtime.memcpy_h2d(dest_id, src, px, py, w, h, elem_per_pe, ...) in SdkRuntime API.
  }];

  let arguments = (ins CSLRuntime_RuntimeType:$runtime,
                       I32Attr:$dest_id,
                       StrAttr:$src_name,
                       IndexAttr:$px,
                       IndexAttr:$py,
                       IndexAttr:$w,
                       IndexAttr:$h,
                       IndexAttr:$elem_per_pe);
  let results = (outs CSLRuntime_RuntimeType:$result);
  let assemblyFormat = "$runtime $dest_id $src_name `at` `(` $px `,` $py `)`"
                       " `with_size` `(` $w `,` $h `)` `elem_per_pe` $elem_per_pe "
                       "attr-dict `:` type($runtime) `->` type($result)";
}

//===----------------------------------------------------------------------===//
// csl_rt.memcpy_d2h
//===----------------------------------------------------------------------===//
// Device-to-host memcpy.
// Maps to runtime.memcpy_d2h(dest, src_id, px, py, w, h, elem_per_pe, ...).
def CSLRuntime_MemcpyD2hOp : CSLRuntime_Op<"memcpy_d2h"> {
  let summary = "Device-to-host memcpy";
  let description = [{
    Performs a device-to-host memcpy. Maps to
    runtime.memcpy_d2h(dest, src_id, px, py, w, h, elem_per_pe, ...) in SdkRuntime API.
  }];

  let arguments = (ins CSLRuntime_RuntimeType:$runtime,
                       StrAttr:$dest_name,
                       I32Attr:$src_id,
                       IndexAttr:$px,
                       IndexAttr:$py,
                       IndexAttr:$w,
                       IndexAttr:$h,
                       IndexAttr:$elem_per_pe);
  let results = (outs CSLRuntime_RuntimeType:$result);
  let assemblyFormat = "$runtime $dest_name `from` `(` $px `,` $py `)`"
                       " `with_size` `(` $w `,` $h `)` `elem_per_pe` $elem_per_pe "
                       "attr-dict `:` type($runtime) `->` type($result)";
}

//===----------------------------------------------------------------------===//
// csl_rt.launch
//===----------------------------------------------------------------------===//
// Launch a host-callable function.
// Maps to runtime.launch(symbol_name, nonblock=False).
def CSLRuntime_LaunchOp : CSLRuntime_Op<"launch"> {
  let summary = "Launch a host-callable function";
  let description = [{
    Launches a host-callable function on the device. Maps to
    runtime.launch("symbol_name", nonblock=False) in SdkRuntime API.
  }];

  let arguments = (ins CSLRuntime_RuntimeType:$runtime,
                       StrAttr:$symbol_name,
                       OptionalAttr<BoolAttr>:$nonblock);
  let results = (outs CSLRuntime_RuntimeType:$result);
  let assemblyFormat = "$runtime $symbol_name attr-dict `:` type($runtime) `->` type($result)";
}

#endif // CSLRUNTIME_OPS
