//===- ChipletAttrs.td - Chiplet attributes ----------------*- tablegen -*-===//
#ifndef CHIPLET_ATTRS
#define CHIPLET_ATTRS

include "chiplet/Dialect/ChipletDialect.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/AttrTypeBase.td"

//===----------------------------------------------------------------------===//
// ChipletScope enum: Fleet's four-level task/memory hierarchy (Table 3).
//===----------------------------------------------------------------------===//

def Chiplet_Scope_Wavefront : I32EnumAttrCase<"Wavefront", 0, "wavefront">;
def Chiplet_Scope_CU        : I32EnumAttrCase<"CU",        1, "cu">;
def Chiplet_Scope_Chiplet   : I32EnumAttrCase<"Chiplet",   2, "chiplet">;
def Chiplet_Scope_Device    : I32EnumAttrCase<"Device",    3, "device">;

def Chiplet_ScopeEnum : I32EnumAttr<"Scope", "Chiplet task/memory scope",
    [Chiplet_Scope_Wavefront, Chiplet_Scope_CU,
     Chiplet_Scope_Chiplet,   Chiplet_Scope_Device]> {
  let cppNamespace = "::mlir::chiplet";
  let genSpecializedAttr = 0;
}

def Chiplet_ScopeAttr : EnumAttr<Chiplet_Dialect, Chiplet_ScopeEnum, "scope"> {
  let assemblyFormat = "`<` $value `>`";
}

//===----------------------------------------------------------------------===//
// CacheModifier enum: Fleet §4.1 three-tier policy.
//===----------------------------------------------------------------------===//

def Chiplet_Cache_All         : I32EnumAttrCase<"CacheAll",    0, "cache_all">;
def Chiplet_Cache_Streaming   : I32EnumAttrCase<"Streaming",   1, "streaming">;
def Chiplet_Cache_NonTemporal : I32EnumAttrCase<"NonTemporal", 2, "non_temporal">;

def Chiplet_CacheModifierEnum : I32EnumAttr<"CacheModifier",
    "Cache allocation policy on load/store",
    [Chiplet_Cache_All, Chiplet_Cache_Streaming, Chiplet_Cache_NonTemporal]> {
  let cppNamespace = "::mlir::chiplet";
  let genSpecializedAttr = 0;
}

def Chiplet_CacheModifierAttr
    : EnumAttr<Chiplet_Dialect, Chiplet_CacheModifierEnum, "cache"> {
  let assemblyFormat = "`<` $value `>`";
}

//===----------------------------------------------------------------------===//
// ChipletTarget: module-level hardware description.
//===----------------------------------------------------------------------===//

def Chiplet_TargetAttr : Chiplet_Attr<"Target", "target"> {
  let summary = "Chiplet target hardware description";
  let description = [{
    A structured attribute attached to the top-level module carrying the
    number of chiplets (XCDs), workers per chiplet (CUs minus scheduler),
    and L2 capacity in bytes. Read by passes; not queried by verifiers in
    the MVP.
  }];

  let parameters = (ins
    "int64_t":$num_chiplets,
    "int64_t":$workers_per_chiplet,
    "int64_t":$l2_capacity_bytes
  );

  let assemblyFormat = [{
    `<` `num_chiplets` `=` $num_chiplets `,`
        `workers_per_chiplet` `=` $workers_per_chiplet `,`
        `l2_capacity_bytes` `=` $l2_capacity_bytes `>`
  }];
}

#endif // CHIPLET_ATTRS
