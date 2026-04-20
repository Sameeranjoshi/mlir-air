//===- ChipletDialect.td - Chiplet dialect -----------------*- tablegen -*-===//
#ifndef CHIPLET_DIALECT
#define CHIPLET_DIALECT

include "mlir/IR/OpBase.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/EnumAttr.td"

def Chiplet_Dialect : Dialect {
  let name = "chiplet";
  let summary = "Dialect for chiplet-aware compilation on multi-die accelerators.";
  let description = [{
    The `chiplet` dialect exposes memory-scope-aware constructs for compiling
    tensor programs onto chiplet-based GPUs (e.g. AMD MI300X/MI350 with 8 XCDs).
    Built after Fleet (ASPLOS 2027 submission): task levels and hierarchical
    synchronization are first-class IR constructs rather than implicit runtime
    behavior.

    Scope enum has four conflated levels (wavefront / cu / chiplet / device)
    following Fleet's task hierarchy (Table 3). Partition identity is symbolic
    via `chiplet.partition_id`, similar to `air.launch`. L2 capacity and other
    hardware parameters live on a module-level `chiplet.target` attribute.
  }];
  let cppNamespace = "::mlir::chiplet";

  let useDefaultAttributePrinterParser = 1;
  let useDefaultTypePrinterParser = 1;

  let extraClassDeclaration = [{
    void registerTypes();
    void registerAttributes();
  }];
}

class Chiplet_Op<string mnemonic, list<Trait> traits = []> :
    Op<Chiplet_Dialect, mnemonic, traits>;

class Chiplet_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<Chiplet_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

class Chiplet_Attr<string name, string attrMnemonic, list<Trait> traits = []>
    : AttrDef<Chiplet_Dialect, name, traits> {
  let mnemonic = attrMnemonic;
}

#endif // CHIPLET_DIALECT
