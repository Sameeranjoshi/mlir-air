//===- CSLDataMovementOps.td - CSL data movement ops ------*- tablegen -*-===//
//
// Part of the air-to-csl project.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#ifndef CSL_DATAMOVEMENT_OPS
#define CSL_DATAMOVEMENT_OPS

include "air/Dialect/CSL/CSLBase.td"

def CSL_GetMemDsdOp : CSL_Op<"get_mem_dsd", []> {
  let summary = "Create a memory-backed Data Structure Descriptor";
  let description = [{
    Creates a DSD referencing a contiguous region of PE-local memory.
    Corresponds to `@get_dsd(mem1d_dsd, ...)` in CSL.

    Example:
    ```mlir
    %dsd = csl.get_mem_dsd %buf, %len : memref<1024xf32>, index -> !csl.dsd
    ```
  }];

  let arguments = (ins
    AnyMemRef:$buffer,
    Index:$length
  );
  let results = (outs CSL_DsdType:$result);

  let assemblyFormat = [{
    $buffer `,` $length `:` type($buffer) `,` type($length) `->` type($result)
    attr-dict
  }];
}

def CSL_GetFabDsdOp : CSL_Op<"get_fab_dsd", []> {
  let summary = "Create a fabric-backed Data Structure Descriptor";
  let description = [{
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
  }];

  let arguments = (ins
    CSL_DsdKindEnum:$kind,
    CSL_ColorType:$color,
    Index:$length
  );
  let results = (outs CSL_DsdType:$result);

  let assemblyFormat = [{
    $kind $color `,` $length `:` type($color) `,` type($length) `->` type($result)
    attr-dict
  }];
}

def CSL_MovOp : CSL_Op<"mov", []> {
  let summary = "Bulk DSD-to-DSD data move";
  let description = [{
    Performs a hardware-accelerated bulk data move between two DSDs.
    Corresponds to `@mov32` or `@mov16` builtins in CSL.

    Example:
    ```mlir
    csl.mov %dst_dsd, %src_dsd : !csl.dsd, !csl.dsd
    ```
  }];

  let arguments = (ins
    CSL_DsdType:$dst,
    CSL_DsdType:$src
  );

  let assemblyFormat = "$dst `,` $src `:` type($dst) `,` type($src) attr-dict";
}

#endif // CSL_DATAMOVEMENT_OPS
