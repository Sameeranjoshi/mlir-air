//===- CSLV2ToPy.cpp - CSL v2 dialect → text/Python emitters ---*- C++ -*-===//
//
// Part of the air-to-csl project.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
//
// Three TranslateFromMLIRRegistration entries that consume a csl.wafer op:
//
//   --emit-csl-program  → pe_program.csl  (CSL text, from csl.program)
//   --emit-csl-layout   → csl_layout.py   (Python sdkLayout API)
//   --emit-csl-host     → run.py          (Python SdkRuntime API)
//
//===----------------------------------------------------------------------===//

#include "air/Dialect/CSL/CSLDialect.h"
#include "air/Dialect/CSL/CSLOps.h"
#include "air/Dialect/CSL/CSLLayoutOps.h"
#include "air/Dialect/CSL/CSLHostOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Tools/mlir-translate/Translation.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/raw_ostream.h"

#include <string>

using namespace mlir;

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

namespace {

/// Return the CSL primitive type name for an MLIR type.
static std::string cslTypeName(Type t) {
  if (t.isF32()) return "f32";
  if (t.isF16()) return "f16";
  if (t.isInteger(32)) return "i32";
  if (t.isInteger(16)) return "i16";
  if (t.isIndex()) return "u16";
  return "f32"; // fallback
}

/// Write N spaces of indentation (2 spaces per level).
static void indent(llvm::raw_ostream &os, unsigned level) {
  for (unsigned i = 0; i < level; ++i)
    os << "  ";
}

/// Resolve a value's string representation from nameMap, or return "?".
static std::string resolve(const llvm::DenseMap<Value, std::string> &nameMap,
                           Value v) {
  auto it = nameMap.find(v);
  if (it != nameMap.end())
    return it->second;
  return "?";
}

/// Emit CSL function body ops into `os`. `outerMap` maps values defined
/// *outside* the function (e.g. csl.var results) to their names.
static LogicalResult emitFuncBody(
    Region &bodyRegion, llvm::raw_ostream &os, unsigned indentLevel,
    const llvm::DenseMap<Value, std::string> &outerMap,
    llvm::DenseMap<Value, std::string> &nameMap, unsigned &tempCount) {

  for (Block &block : bodyRegion) {
    for (Operation &op : block) {
      // csl.return — implicit in CSL
      if (isa<xilinx::csl::ReturnOp>(&op))
        continue;

      // scf.yield — no emission
      if (isa<scf::YieldOp>(&op))
        continue;

      // arith.constant
      if (auto constOp = dyn_cast<arith::ConstantOp>(&op)) {
        if (auto intAttr = dyn_cast<IntegerAttr>(constOp.getValue()))
          nameMap[constOp.getResult()] = std::to_string(intAttr.getInt());
        else if (auto floatAttr = dyn_cast<FloatAttr>(constOp.getValue()))
          nameMap[constOp.getResult()] =
              std::to_string(floatAttr.getValueAsDouble());
        continue;
      }

      // arith.addf / arith.addi
      if (auto addOp = dyn_cast<arith::AddFOp>(&op)) {
        std::string lhs = resolve(nameMap, addOp.getLhs());
        std::string rhs = resolve(nameMap, addOp.getRhs());
        std::string tname = "t" + std::to_string(tempCount++);
        indent(os, indentLevel);
        os << "var " << tname << ": f32 = " << lhs << " + " << rhs << ";\n";
        nameMap[addOp.getResult()] = tname;
        continue;
      }
      if (auto addOp = dyn_cast<arith::AddIOp>(&op)) {
        std::string lhs = resolve(nameMap, addOp.getLhs());
        std::string rhs = resolve(nameMap, addOp.getRhs());
        std::string tname = "t" + std::to_string(tempCount++);
        indent(os, indentLevel);
        os << "var " << tname << ": i32 = " << lhs << " + " << rhs << ";\n";
        nameMap[addOp.getResult()] = tname;
        continue;
      }

      // memref.load
      if (auto loadOp = dyn_cast<memref::LoadOp>(&op)) {
        // Resolve buffer name — prefer outer (csl.var) map
        Value memref = loadOp.getMemref();
        std::string bufName = resolve(outerMap, memref);
        if (bufName == "?")
          bufName = resolve(nameMap, memref);
        std::string idxName;
        if (!loadOp.getIndices().empty())
          idxName = resolve(nameMap, loadOp.getIndices()[0]);
        std::string tname = "t" + std::to_string(tempCount++);
        indent(os, indentLevel);
        os << "var " << tname << " = " << bufName << "[" << idxName << "];\n";
        nameMap[loadOp.getResult()] = tname;
        continue;
      }

      // memref.store
      if (auto storeOp = dyn_cast<memref::StoreOp>(&op)) {
        Value memref = storeOp.getMemref();
        std::string bufName = resolve(outerMap, memref);
        if (bufName == "?")
          bufName = resolve(nameMap, memref);
        std::string idxName;
        if (!storeOp.getIndices().empty())
          idxName = resolve(nameMap, storeOp.getIndices()[0]);
        std::string valName = resolve(nameMap, storeOp.getValue());
        indent(os, indentLevel);
        os << bufName << "[" << idxName << "] = " << valName << ";\n";
        continue;
      }

      // scf.for → CSL while (...) : (incr) { body }
      if (auto forOp = dyn_cast<scf::ForOp>(&op)) {
        std::string loStr = resolve(nameMap, forOp.getLowerBound());
        std::string hiStr = resolve(nameMap, forOp.getUpperBound());
        std::string stepStr = resolve(nameMap, forOp.getStep());
        std::string iname = "i" + std::to_string(tempCount++);
        nameMap[forOp.getInductionVar()] = iname;

        indent(os, indentLevel);
        os << "var " << iname << ": u16 = " << loStr << ";\n";
        indent(os, indentLevel);
        os << "while (" << iname << " < " << hiStr << ") : (" << iname
           << " += " << stepStr << ") {\n";

        if (failed(emitFuncBody(forOp.getBodyRegion(), os, indentLevel + 1,
                                outerMap, nameMap, tempCount)))
          return failure();

        indent(os, indentLevel);
        os << "}\n";
        continue;
      }

      // Unknown op
      op.emitOpError("CSLV2ToPy: unsupported op in function body: ");
      return failure();
    }
  }
  return success();
}

//===----------------------------------------------------------------------===//
// ProgramEmitter — emit pe_program.csl from csl.wafer > csl.program
//===----------------------------------------------------------------------===//

class ProgramEmitter {
public:
  explicit ProgramEmitter(llvm::raw_ostream &os) : os(os) {}
  LogicalResult emit(ModuleOp module);

private:
  llvm::raw_ostream &os;
};

LogicalResult ProgramEmitter::emit(ModuleOp module) {
  namespace cslns = xilinx::csl;

  // Find csl.wafer
  cslns::WaferOp wafer;
  module.walk([&](cslns::WaferOp w) { wafer = w; });
  if (!wafer) {
    os << "// No csl.wafer found in module.\n";
    return success();
  }

  // Find the first csl.program
  cslns::ProgramOp prog;
  wafer.walk([&](cslns::ProgramOp p) {
    if (!prog) prog = p;
  });
  if (!prog) {
    os << "// No csl.program found in csl.wafer.\n";
    return success();
  }

  os << "// Generated by air-translate --emit-csl-program\n\n";

  // Emit memcpy module preamble (required for SDK host communication).
  os << "param memcpy_params: comptime_struct;\n";
  os << "const sys_mod = @import_module(\"<memcpy/memcpy>\", memcpy_params);\n\n";

  // 1. Emit param declarations from block args with !csl.comptime<T> type.
  Region &progBody = prog.getBody();
  if (!progBody.empty()) {
    Block &entryBlock = progBody.front();
    for (BlockArgument arg : entryBlock.getArguments()) {
      Type t = arg.getType();
      if (auto comptime = mlir::dyn_cast<cslns::ComptimeType>(t)) {
        // Get the arg name from arg number
        std::string argName = "arg" + std::to_string(arg.getArgNumber());
        os << "param " << argName << ": " << cslTypeName(comptime.getInnerType())
           << ";\n";
      }
    }
    os << "\n";
  }

  // Build outer map: csl.var SSA results → sym_name
  llvm::DenseMap<Value, std::string> outerMap;

  // 2. Emit var declarations
  bool hasVars = false;
  for (Operation &op : prog.getBody().front()) {
    if (auto varOp = dyn_cast<cslns::VarOp>(&op)) {
      auto memTy = dyn_cast<MemRefType>(varOp.getResult().getType());
      if (!memTy || memTy.getRank() != 1) {
        varOp.emitOpError("CSLV2ToPy: csl.var must have 1-D memref type");
        return failure();
      }
      std::string eltName = cslTypeName(memTy.getElementType());
      int64_t dim = memTy.getDimSize(0);
      os << "var " << varOp.getSymName() << ": [" << dim << "]" << eltName
         << ";\n";
      outerMap[varOp.getResult()] = varOp.getSymName().str();
      hasVars = true;
    }
  }
  if (hasVars)
    os << "\n";

  // Emit pointer variables for exported buffers (SDK memcpy requirement).
  // Direction determines mutability: "in" → var (host writes), "out" → const (host reads).
  bool hasPointers = false;
  for (Operation &op : prog.getBody().front()) {
    if (auto expOp = dyn_cast<cslns::ExportOp>(&op)) {
      auto kind = expOp.getKind();
      if (kind.has_value() && *kind == "func")
        continue; // function exports don't need pointers

      auto dir = expOp.getDirection();
      // Skip internal exports — they are not host-visible and don't need pointers.
      if (dir.has_value() && *dir == "internal")
        continue;

      StringRef sym = expOp.getSym();

      // Find the var op to get element type
      std::string eltName = "f32"; // default
      for (Operation &varOp : prog.getBody().front()) {
        if (auto v = dyn_cast<cslns::VarOp>(&varOp)) {
          if (v.getSymName() == sym) {
            if (auto memTy = dyn_cast<MemRefType>(v.getResult().getType()))
              eltName = cslTypeName(memTy.getElementType());
            break;
          }
        }
      }

      // "out" direction means host reads → const pointer
      // "in" direction means host writes → var pointer
      bool isOutput = dir.has_value() && *dir == "out";
      os << (isOutput ? "const " : "var ") << sym << "_ptr: [*]" << eltName
         << " = &" << sym << ";\n";
      hasPointers = true;
    }
  }
  if (hasPointers)
    os << "\n";

  // 3. Emit function definitions
  for (Operation &op : prog.getBody().front()) {
    auto funcOp = dyn_cast<cslns::FuncOp>(&op);
    if (!funcOp) continue;

    os << "fn " << funcOp.getSymName() << "() void {\n";

    llvm::DenseMap<Value, std::string> nameMap;
    // Copy outer map into nameMap (so loads/stores can find var names)
    for (auto &kv : outerMap)
      nameMap[kv.first] = kv.second;

    unsigned tempCount = 0;
    if (failed(emitFuncBody(funcOp.getBody(), os, /*indentLevel=*/1,
                            outerMap, nameMap, tempCount)))
      return failure();

    // Signal host that compute is done (required by SDK memcpy infrastructure).
    os << "  sys_mod.unblock_cmd_stream();\n";
    os << "}\n\n";
  }

  // 4. Emit comptime block for csl.export ops (direction != "internal")
  bool hasExports = false;
  for (Operation &op : prog.getBody().front()) {
    if (auto expOp = dyn_cast<cslns::ExportOp>(&op)) {
      auto dir = expOp.getDirection();
      auto kind = expOp.getKind();
      // kind="func" exports are always emitted — they represent host-callable
      // functions and must appear in the comptime block regardless of direction.
      // We check the kind attribute rather than walking the symbol table because
      // the attribute is the authoritative contract (see csl.export in CSLOps.td).
      bool isFuncExport = kind.has_value() && *kind == "func";
      if (!isFuncExport && dir.has_value() && *dir == "internal")
        continue;
      if (!hasExports) {
        os << "comptime {\n";
        hasExports = true;
      }
      // sym is a FlatSymbolRefAttr → getSym() returns StringRef
      StringRef sym = expOp.getSym();
      if (isFuncExport) {
        // Function exports: no pointer, no alias string needed
        os << "  @export_symbol(" << sym << ");\n";
      } else {
        // Buffer exports: export the pointer variable with alias
        StringRef alias = expOp.getAlias().value_or(sym);
        os << "  @export_symbol(" << sym << "_ptr, \"" << alias << "\");\n";
      }
    }
  }
  if (hasExports)
    os << "}\n";

  return success();
}

//===----------------------------------------------------------------------===//
// LayoutEmitter — emit csl_layout.py from csl.wafer > csl.layout
//===----------------------------------------------------------------------===//

class LayoutEmitter {
public:
  explicit LayoutEmitter(llvm::raw_ostream &os) : os(os) {}
  LogicalResult emit(ModuleOp module);

private:
  llvm::raw_ostream &os;
};

LogicalResult LayoutEmitter::emit(ModuleOp module) {
  namespace cslns = xilinx::csl;
  namespace layoutns = xilinx::csl_layout;

  // Find csl.wafer
  cslns::WaferOp wafer;
  module.walk([&](cslns::WaferOp w) { wafer = w; });
  if (!wafer) {
    os << "# No csl.wafer found in module.\n";
    return success();
  }

  // Find csl.layout
  cslns::LayoutOp layout;
  for (Operation &op : wafer.getBody().front()) {
    if (auto l = dyn_cast<cslns::LayoutOp>(&op)) {
      layout = l;
      break;
    }
  }
  if (!layout) {
    os << "# No csl.layout found in csl.wafer.\n";
    return success();
  }

  // Find the first csl.program to get the program name
  std::string progName = "pe_program";
  for (Operation &op : wafer.getBody().front()) {
    if (auto prog = dyn_cast<cslns::ProgramOp>(&op)) {
      progName = prog.getSymName().str();
      break;
    }
  }

  // Collect place ops to determine grid dimensions
  int64_t maxX = 0, maxY = 0;
  bool hasPlace = false;

  // Collect compile-time param bindings from place ops
  // (extra attrs on csl_layout.place beyond prog/px/py)
  struct ParamBinding {
    std::string name;
    int64_t value;
  };
  llvm::SmallVector<ParamBinding, 4> params;

  layout.getBody().walk([&](layoutns::PlaceOp placeOp) {
    int64_t px = placeOp.getPx();
    int64_t py = placeOp.getPy();
    if (px > maxX) maxX = px;
    if (py > maxY) maxY = py;
    hasPlace = true;

    // Walk extra attributes (not prog, px, py)
    for (NamedAttribute attr : placeOp->getAttrs()) {
      StringRef name = attr.getName().getValue();
      if (name == "prog" || name == "px" || name == "py")
        continue;
      if (auto intAttr = dyn_cast<IntegerAttr>(attr.getValue()))
        params.push_back({name.str(), intAttr.getInt()});
    }
  });

  int64_t W = hasPlace ? maxX + 1 : 1;
  int64_t H = hasPlace ? maxY + 1 : 1;

  os << "# Generated by air-translate --emit-csl-layout\n\n";
  os << "from cerebras.sdk.runtime.sdkruntimepybind import SdkLayout, SdkTarget\n\n";
  os << "def get_layout(target: SdkTarget) -> SdkLayout:\n";
  os << "    layout = SdkLayout(target)\n";
  os << "    region = layout.create_code_region(\"" << progName << ".csl\", \""
     << progName << "\", " << W << ", " << H << ")\n";

  for (auto &p : params)
    os << "    region.set_param_all(\"" << p.name << "\", " << p.value << ")\n";

  os << "    region.place(0, 0)\n";
  os << "    return layout\n";

  return success();
}

//===----------------------------------------------------------------------===//
// HostEmitter — emit run.py from csl.wafer > csl.host
//===----------------------------------------------------------------------===//

class HostEmitter {
public:
  explicit HostEmitter(llvm::raw_ostream &os) : os(os) {}
  LogicalResult emit(ModuleOp module);

private:
  llvm::raw_ostream &os;
};

LogicalResult HostEmitter::emit(ModuleOp module) {
  namespace cslns = xilinx::csl;
  namespace hostns = xilinx::csl_host;

  // Find csl.wafer
  cslns::WaferOp wafer;
  module.walk([&](cslns::WaferOp w) { wafer = w; });
  if (!wafer) {
    os << "# No csl.wafer found in module.\n";
    return success();
  }

  // Find csl.host
  cslns::HostOp host;
  for (Operation &op : wafer.getBody().front()) {
    if (auto h = dyn_cast<cslns::HostOp>(&op)) {
      host = h;
      break;
    }
  }
  if (!host) {
    os << "# No csl.host found in csl.wafer.\n";
    return success();
  }

  // Collect block arg names from csl.host region
  // The region's entry block carries the function parameters.
  llvm::SmallVector<std::string, 4> argNames;
  llvm::SmallVector<Type, 4> argTypes;
  if (!host.getBody().empty()) {
    Block &entry = host.getBody().front();
    for (BlockArgument arg : entry.getArguments()) {
      argNames.push_back("arg" + std::to_string(arg.getArgNumber()));
      argTypes.push_back(arg.getType());
    }
  }

  // Determine N from the first memref arg
  int64_t N = -1;
  for (Type t : argTypes) {
    if (auto memTy = dyn_cast<MemRefType>(t)) {
      if (memTy.getRank() >= 1 && memTy.getDimSize(0) > 0) {
        N = memTy.getDimSize(0);
        break;
      }
    }
  }

  // Build function signature
  std::string sig = "target";
  for (auto &name : argNames)
    sig += ", " + name;

  os << "# Generated by air-translate --emit-csl-host\n\n";
  os << "from csl_layout import get_layout\n";
  os << "from cerebras.sdk.client import SdkRuntime\n";
  os << "import numpy as np\n\n";

  os << "def main(" << sig << "):\n";
  os << "    artifacts = get_layout(target).compile(\"out/\")\n";
  if (N > 0)
    os << "    N = " << N << "\n";
  os << "    with SdkRuntime(artifacts) as runner:\n";

  // Walk csl_host ops in order
  if (!host.getBody().empty()) {
    Block &entry = host.getBody().front();
    for (Operation &op : entry) {
      // csl_host.memcpy_h2d
      if (auto h2dOp = dyn_cast<hostns::MemcpyH2DOp>(&op)) {
        // sym is a SymbolRefAttr — get leaf name
        SymbolRefAttr symAttr = h2dOp.getSym();
        StringRef leafSym;
        if (!symAttr.getNestedReferences().empty())
          leafSym = symAttr.getNestedReferences().back().getValue();
        else
          leafSym = symAttr.getRootReference().getValue();

        // Resolve host arg name for src
        Value src = h2dOp.getSrc();
        std::string srcName = "?";
        if (auto barg = dyn_cast<BlockArgument>(src))
          srcName = argNames[barg.getArgNumber()];

        int64_t px = h2dOp.getPx();
        int64_t py = h2dOp.getPy();
        int64_t w = h2dOp.getWidth();
        int64_t h = h2dOp.getHeight();
        std::string nStr = (N > 0) ? "N" : "len(" + srcName + ")";

        os << "        runner.memcpy_h2d(runner.get_id(\"" << leafSym
           << "\"), " << srcName << ", " << px << ", " << py << ", " << w
           << ", " << h << ", " << nStr << ")\n";
        continue;
      }

      // csl_host.memcpy_d2h
      if (auto d2hOp = dyn_cast<hostns::MemcpyD2HOp>(&op)) {
        SymbolRefAttr symAttr = d2hOp.getSym();
        StringRef leafSym;
        if (!symAttr.getNestedReferences().empty())
          leafSym = symAttr.getNestedReferences().back().getValue();
        else
          leafSym = symAttr.getRootReference().getValue();

        Value dst = d2hOp.getDst();
        std::string dstName = "?";
        if (auto barg = dyn_cast<BlockArgument>(dst))
          dstName = argNames[barg.getArgNumber()];

        int64_t px = d2hOp.getPx();
        int64_t py = d2hOp.getPy();
        int64_t w = d2hOp.getWidth();
        int64_t h = d2hOp.getHeight();
        std::string nStr = (N > 0) ? "N" : "len(" + dstName + ")";

        os << "        runner.memcpy_d2h(" << dstName
           << ", runner.get_id(\"" << leafSym << "\"), " << px << ", " << py
           << ", " << w << ", " << h << ", " << nStr << ")\n";
        continue;
      }

      // csl_host.launch
      if (auto launchOp = dyn_cast<hostns::LaunchOp>(&op)) {
        SymbolRefAttr symAttr = launchOp.getSym();
        StringRef leafSym;
        if (!symAttr.getNestedReferences().empty())
          leafSym = symAttr.getNestedReferences().back().getValue();
        else
          leafSym = symAttr.getRootReference().getValue();

        os << "        runner.launch(\"" << leafSym
           << "\", nonblock=False)\n";
        continue;
      }
    }
  }

  return success();
}

} // namespace

//===----------------------------------------------------------------------===//
// Registration
//===----------------------------------------------------------------------===//

namespace xilinx {
namespace csl {

void registerCSLV2ToPyTranslations() {
  // --emit-csl-program
  TranslateFromMLIRRegistration regProgram(
      "emit-csl-program", "Emit CSL v2 PE program source (pe_program.csl)",
      [](ModuleOp module, llvm::raw_ostream &os) -> LogicalResult {
        ProgramEmitter emitter(os);
        return emitter.emit(module);
      },
      [](DialectRegistry &registry) {
        registry.insert<xilinx::csl::CSLDialect,
                        xilinx::csl_layout::CSLLayoutDialect,
                        xilinx::csl_host::CSLHostDialect,
                        mlir::arith::ArithDialect,
                        mlir::func::FuncDialect,
                        mlir::memref::MemRefDialect,
                        mlir::scf::SCFDialect>();
      });

  // --emit-csl-layout
  TranslateFromMLIRRegistration regLayout(
      "emit-csl-layout", "Emit CSL v2 layout Python (csl_layout.py)",
      [](ModuleOp module, llvm::raw_ostream &os) -> LogicalResult {
        LayoutEmitter emitter(os);
        return emitter.emit(module);
      },
      [](DialectRegistry &registry) {
        registry.insert<xilinx::csl::CSLDialect,
                        xilinx::csl_layout::CSLLayoutDialect,
                        xilinx::csl_host::CSLHostDialect,
                        mlir::arith::ArithDialect,
                        mlir::func::FuncDialect,
                        mlir::memref::MemRefDialect,
                        mlir::scf::SCFDialect>();
      });

  // --emit-csl-host
  TranslateFromMLIRRegistration regHost(
      "emit-csl-host", "Emit CSL v2 host runtime Python (run.py)",
      [](ModuleOp module, llvm::raw_ostream &os) -> LogicalResult {
        HostEmitter emitter(os);
        return emitter.emit(module);
      },
      [](DialectRegistry &registry) {
        registry.insert<xilinx::csl::CSLDialect,
                        xilinx::csl_layout::CSLLayoutDialect,
                        xilinx::csl_host::CSLHostDialect,
                        mlir::arith::ArithDialect,
                        mlir::func::FuncDialect,
                        mlir::memref::MemRefDialect,
                        mlir::scf::SCFDialect>();
      });
}

} // namespace csl
} // namespace xilinx
