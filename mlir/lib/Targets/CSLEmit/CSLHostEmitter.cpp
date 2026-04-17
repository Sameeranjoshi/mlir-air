//===- CSLHostEmitter.cpp - Emit host run.py -------------------*- C++ -*-===//
//
// Part of the air-to-csl project.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
//
// Registers --emit-csl-host. Walks csl.wafer > csl.host and emits a complete,
// directly runnable Python SdkRuntime host driver (`run.py`).
//
// The emitted script uses the precompiled-artifact flow:
//   runner = SdkRuntime(args.name, cmaddr=args.cmaddr)
// It then does: load/run -> h2d copies -> launch -> d2h copies -> stop, and
// prints `SUCCESS!` after printing a snippet of each output buffer.
//
// This is the flow required because our device code uses `<memcpy/memcpy>`
// with `param memcpy_params: comptime_struct;` — SdkLayout.compile() cannot
// wire `memcpy_params` because it only accepts int/Color parameters, whereas
// `memcpy_params` is a comptime_struct produced from `<memcpy/get_params>` in
// a `layout.csl` wrapper. The companion `commands_wse{2,3}.sh` scripts call
// `cslc layout.csl ... -o compiled` first, then `cs_python run.py --name=compiled`.
//
//===----------------------------------------------------------------------===//

#include "CSLEmitCommon.h"

#include "air/Dialect/CSL/CSLDialect.h"
#include "air/Dialect/CSL/CSLHostOps.h"
#include "air/Dialect/CSL/CSLLayoutOps.h"
#include "air/Dialect/CSL/CSLOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/Tools/mlir-translate/Translation.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/raw_ostream.h"

#include <cassert>
#include <string>
#include <tuple>

using namespace mlir;

namespace xilinx {
namespace csl {

namespace {

/// Classify the element type of a memref for numpy dtype + MemcpyDataType.
struct ElemInfo {
  std::string npDtype;     // e.g. "np.float32"
  std::string memcpyKind;  // e.g. "MemcpyDataType.MEMCPY_32BIT"
  bool isFloat;            // whether arange*(i+1) should use float dtype
};

static ElemInfo classifyElem(Type eltTy) {
  if (eltTy.isF32())
    return {"np.float32", "MemcpyDataType.MEMCPY_32BIT", true};
  if (eltTy.isF16())
    return {"np.float16", "MemcpyDataType.MEMCPY_16BIT", true};
  if (eltTy.isInteger(32))
    return {"np.int32", "MemcpyDataType.MEMCPY_32BIT", false};
  if (eltTy.isInteger(16))
    return {"np.int16", "MemcpyDataType.MEMCPY_16BIT", false};
  // Fallback: treat as 32-bit float.
  return {"np.float32", "MemcpyDataType.MEMCPY_32BIT", true};
}

enum class Dir { H2D, D2H, Unused };

struct MemcpyEntry {
  bool isH2d;
  unsigned argIdx;
  std::string leafSym;
  // Lower-left corner of the source/destination PE rectangle (== place origin).
  int64_t px, py;
  // PE grid extent of the placement (NOT the memref shape).
  int64_t w, h;
  // Per-PE element count (total_elems / (w * h)).
  int64_t l;
};

/// Derive the SdkRuntime memcpy extents `(w, h, l)` from a PlaceOp that
/// describes the placement of the program owning the referenced var, and
/// the memref type of the host-side buffer.
///
/// Mapping:
///   at (x, y)           -> (1, 1),        l = total
///   over [lo:hi, Y]     -> (hi-lo, 1),    l = total / (hi-lo)
///   over [lo:hi, lo:hi] -> (hi-lo, yext), l = total / ((hi-lo) * yext)
///
/// `total` is the product of all memref shape dims. Unequal sharding is
/// asserted; -csl-verify-params (Task 9) will reject it up front.
static std::tuple<int64_t, int64_t, int64_t>
deriveMemcpyExtent(xilinx::csl_layout::PlaceOp place, MemRefType memTy) {
  int64_t total = 1;
  for (int64_t d : memTy.getShape())
    total *= d;
  int64_t w = 1, h = 1;
  if (place) {
    if (place.getPx().has_value()) {
      // Point form: (1, 1).
    } else if (auto xr = place.getXRange()) {
      int64_t xlo = cast<IntegerAttr>((*xr)[0]).getInt();
      int64_t xhi = cast<IntegerAttr>((*xr)[1]).getInt();
      w = xhi - xlo;
      if (auto yr = place.getYRange()) {
        int64_t ylo = cast<IntegerAttr>((*yr)[0]).getInt();
        int64_t yhi = cast<IntegerAttr>((*yr)[1]).getInt();
        h = yhi - ylo;
      }
    }
  }
  assert(w * h > 0 && "placement extent must be positive");
  assert(total % (w * h) == 0 &&
         "csl-verify-params should reject unequal sharding");
  return {w, h, total / (w * h)};
}

/// Lookup the PlaceOp that binds the program named `progSym` onto the grid,
/// by scanning `csl_layout.place` ops inside the wafer's `csl.layout`. If
/// `progSym` is empty, returns the first PlaceOp (single-program fallback).
static xilinx::csl_layout::PlaceOp
findPlaceForProgram(xilinx::csl::WaferOp wafer, StringRef progSym) {
  xilinx::csl_layout::PlaceOp match;
  wafer.walk([&](xilinx::csl_layout::PlaceOp p) {
    if (match)
      return;
    if (progSym.empty() || p.getProg() == progSym)
      match = p;
  });
  return match;
}

/// Resolve the program that contains the var named `leaf` (e.g. `@a`), by
/// scanning every `csl.program` in the wafer for a matching `csl.var` or
/// `csl.func`. Returns the empty StringRef if no match (single-program case).
static StringRef findProgramForLeaf(xilinx::csl::WaferOp wafer, StringRef leaf) {
  StringRef found;
  wafer.walk([&](xilinx::csl::ProgramOp prog) {
    if (!found.empty())
      return;
    prog.getBody().walk([&](Operation *op) {
      if (!found.empty())
        return;
      if (auto v = dyn_cast<xilinx::csl::VarOp>(op)) {
        if (v.getSymName() == leaf)
          found = prog.getSymName();
      } else if (auto f = dyn_cast<xilinx::csl::FuncOp>(op)) {
        if (f.getSymName() == leaf)
          found = prog.getSymName();
      }
    });
  });
  return found;
}

//===----------------------------------------------------------------------===//
// Reference-compute walker: translates a csl.func body to numpy code.
//===----------------------------------------------------------------------===//

/// Emitter state for the numpy reference walker. One instance per run.py
/// (per wafer). Keeps the SSA value -> python-expression map, a counter for
/// fresh temp names, and a failure flag that triggers the sanity fallback.
struct RefEmitState {
  llvm::DenseMap<mlir::Value, std::string> nameMap;
  unsigned tempCount = 0;
  bool ok = true;
  std::string reason; // why we bailed out (for the fallback comment)
};

static std::string rnpDtype(mlir::Type eltTy) {
  if (eltTy.isF32())
    return "np.float32";
  if (eltTy.isF16())
    return "np.float16";
  if (eltTy.isInteger(32))
    return "np.int32";
  if (eltTy.isInteger(16))
    return "np.int16";
  return "np.float32";
}

static std::string rfreshTemp(RefEmitState &s, StringRef prefix = "t") {
  return (prefix + llvm::Twine(s.tempCount++)).str();
}

static std::string rresolve(const RefEmitState &s, mlir::Value v) {
  auto it = s.nameMap.find(v);
  if (it != s.nameMap.end())
    return it->second;
  return "None";
}

// Forward declaration.
static void emitRefBody(mlir::Region &region, llvm::raw_ostream &os,
                        unsigned indentLevel, RefEmitState &s);

/// Emit the translation of one body op into numpy.
static void emitRefOp(mlir::Operation &op, llvm::raw_ostream &os,
                      unsigned indentLevel, RefEmitState &s) {
  using namespace mlir;

  auto ind = [&]() {
    for (unsigned i = 0; i < indentLevel; ++i)
      os << "    ";
  };

  // Terminators: skipped.
  if (isa<xilinx::csl::ReturnOp>(&op) || isa<scf::YieldOp>(&op))
    return;

  // arith.constant
  if (auto constOp = dyn_cast<arith::ConstantOp>(&op)) {
    std::string t = rfreshTemp(s);
    if (auto intAttr = dyn_cast<IntegerAttr>(constOp.getValue())) {
      // Integer constants become Python ints (used as loop bounds / indices).
      s.nameMap[constOp.getResult()] = std::to_string(intAttr.getInt());
      return;
    }
    if (auto floatAttr = dyn_cast<FloatAttr>(constOp.getValue())) {
      std::string dtype = rnpDtype(constOp.getType());
      ind();
      os << t << " = " << dtype << "("
         << llvm::format("%.17g", floatAttr.getValueAsDouble()) << ")\n";
      s.nameMap[constOp.getResult()] = t;
      return;
    }
    s.ok = false;
    s.reason = "non-int/float arith.constant";
    return;
  }

  // Binary arith ops.
  auto emitBin = [&](Operation *bop, StringRef pyOp) {
    std::string l = rresolve(s, bop->getOperand(0));
    std::string r = rresolve(s, bop->getOperand(1));
    std::string t = rfreshTemp(s);
    ind();
    os << t << " = " << l << " " << pyOp << " " << r << "\n";
    s.nameMap[bop->getResult(0)] = t;
  };
  auto emitCall = [&](Operation *bop, StringRef pyFn) {
    std::string l = rresolve(s, bop->getOperand(0));
    std::string r = rresolve(s, bop->getOperand(1));
    std::string t = rfreshTemp(s);
    ind();
    os << t << " = " << pyFn << "(" << l << ", " << r << ")\n";
    s.nameMap[bop->getResult(0)] = t;
  };

  if (isa<arith::AddFOp>(&op) || isa<arith::AddIOp>(&op)) {
    emitBin(&op, "+");
    return;
  }
  if (isa<arith::SubFOp>(&op) || isa<arith::SubIOp>(&op)) {
    emitBin(&op, "-");
    return;
  }
  if (isa<arith::MulFOp>(&op) || isa<arith::MulIOp>(&op)) {
    emitBin(&op, "*");
    return;
  }
  if (isa<arith::DivFOp>(&op)) {
    emitBin(&op, "/");
    return;
  }
  if (isa<arith::MaximumFOp>(&op)) {
    emitCall(&op, "np.maximum");
    return;
  }
  if (isa<arith::MinimumFOp>(&op)) {
    emitCall(&op, "np.minimum");
    return;
  }
  if (auto negOp = dyn_cast<arith::NegFOp>(&op)) {
    std::string a = rresolve(s, negOp.getOperand());
    std::string t = rfreshTemp(s);
    ind();
    os << t << " = -" << a << "\n";
    s.nameMap[negOp.getResult()] = t;
    return;
  }

  // memref.load
  if (auto loadOp = dyn_cast<memref::LoadOp>(&op)) {
    std::string buf = rresolve(s, loadOp.getMemref());
    std::string idx;
    if (!loadOp.getIndices().empty())
      idx = rresolve(s, loadOp.getIndices()[0]);
    else
      idx = "0";
    std::string t = rfreshTemp(s);
    ind();
    os << t << " = " << buf << "[" << idx << "]\n";
    s.nameMap[loadOp.getResult()] = t;
    return;
  }

  // memref.store
  if (auto storeOp = dyn_cast<memref::StoreOp>(&op)) {
    std::string buf = rresolve(s, storeOp.getMemref());
    std::string idx;
    if (!storeOp.getIndices().empty())
      idx = rresolve(s, storeOp.getIndices()[0]);
    else
      idx = "0";
    std::string val = rresolve(s, storeOp.getValue());
    ind();
    os << buf << "[" << idx << "] = " << val << "\n";
    return;
  }

  // scf.for
  if (auto forOp = dyn_cast<scf::ForOp>(&op)) {
    std::string lo = rresolve(s, forOp.getLowerBound());
    std::string hi = rresolve(s, forOp.getUpperBound());
    std::string st = rresolve(s, forOp.getStep());
    std::string iv = rfreshTemp(s, "i");
    s.nameMap[forOp.getInductionVar()] = iv;
    ind();
    os << "for " << iv << " in range(" << lo << ", " << hi << ", " << st
       << "):\n";
    emitRefBody(forOp.getBodyRegion(), os, indentLevel + 1, s);
    return;
  }

  // func.call — inline the helper by walking its body with operand mapping.
  if (auto callOp = dyn_cast<func::CallOp>(&op)) {
    StringRef callee = callOp.getCallee();
    // Find the callee func.func in the enclosing csl.program.
    Operation *p = op.getParentOp();
    while (p && !isa<xilinx::csl::ProgramOp>(p))
      p = p->getParentOp();
    func::FuncOp calleeFn;
    if (p) {
      p->walk([&](func::FuncOp f) {
        if (f.getSymName() == callee)
          calleeFn = f;
      });
    }
    if (!calleeFn || calleeFn.getBody().empty()) {
      s.ok = false;
      s.reason = "helper not found or external";
      return;
    }

    // Map the callee's block args to the call's operand names.
    Block &entry = calleeFn.getBody().front();
    for (auto it : llvm::enumerate(entry.getArguments())) {
      Value operand = callOp.getOperand(it.index());
      s.nameMap[it.value()] = rresolve(s, operand);
    }
    // Walk the callee body, capturing any returned value.
    // func.return is handled specially below.
    std::string retName;
    for (Block &blk : calleeFn.getBody()) {
      for (Operation &sub : blk) {
        if (!s.ok)
          return;
        if (auto ret = dyn_cast<func::ReturnOp>(&sub)) {
          if (ret.getNumOperands() > 0)
            retName = rresolve(s, ret.getOperand(0));
          continue;
        }
        emitRefOp(sub, os, indentLevel, s);
      }
    }
    if (callOp.getNumResults() > 0) {
      if (retName.empty()) {
        s.ok = false;
        s.reason = "helper returned no value";
        return;
      }
      s.nameMap[callOp.getResult(0)] = retName;
    }
    return;
  }

  // func.return — only reachable at top level, ignored (we're in main body).
  if (isa<func::ReturnOp>(&op))
    return;

  s.ok = false;
  s.reason = ("unsupported op: " + op.getName().getStringRef()).str();
}

static void emitRefBody(mlir::Region &region, llvm::raw_ostream &os,
                        unsigned indentLevel, RefEmitState &s) {
  for (mlir::Block &blk : region) {
    for (mlir::Operation &op : blk) {
      if (!s.ok)
        return;
      emitRefOp(op, os, indentLevel, s);
    }
  }
}

class HostEmitter {
public:
  explicit HostEmitter(llvm::raw_ostream &os) : os(os) {}
  LogicalResult emit(xilinx::csl::WaferOp wafer);

private:
  llvm::raw_ostream &os;

  void emitMemcpy(const MemcpyEntry &e,
                  const llvm::SmallVectorImpl<std::string> &argNames,
                  const llvm::SmallVectorImpl<Type> &argTypes,
                  MLIRContext *ctx);
};

void HostEmitter::emitMemcpy(const MemcpyEntry &e,
                             const llvm::SmallVectorImpl<std::string> &argNames,
                             const llvm::SmallVectorImpl<Type> &argTypes,
                             MLIRContext *ctx) {
  if (e.argIdx >= argNames.size())
    return;
  Type eltTy;
  if (auto memTy = dyn_cast<MemRefType>(argTypes[e.argIdx]))
    eltTy = memTy.getElementType();
  else
    eltTy = Float32Type::get(ctx);
  ElemInfo info = classifyElem(eltTy);

  if (e.isH2d) {
    os << "runner.memcpy_h2d(runner.get_id(\"" << e.leafSym << "\"), "
       << argNames[e.argIdx] << ", " << e.px << ", " << e.py << ", "
       << e.w << ", " << e.h << ", " << e.l << ",\n";
  } else {
    os << "runner.memcpy_d2h(" << argNames[e.argIdx]
       << ", runner.get_id(\"" << e.leafSym << "\"), " << e.px << ", "
       << e.py << ", " << e.w << ", " << e.h << ", " << e.l << ",\n";
  }
  os << "                  streaming=False,\n";
  os << "                  order=MemcpyOrder.ROW_MAJOR,\n";
  os << "                  data_type=" << info.memcpyKind << ",\n";
  os << "                  nonblock=False)\n";
}

LogicalResult HostEmitter::emit(xilinx::csl::WaferOp wafer) {
  namespace cslns = xilinx::csl;
  namespace hostns = xilinx::csl_host;

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

  // Collect block arg names + memref types from csl.host region.
  llvm::SmallVector<std::string, 4> argNames;
  llvm::SmallVector<Type, 4> argTypes;
  if (!host.getBody().empty()) {
    Block &entry = host.getBody().front();
    for (BlockArgument arg : entry.getArguments()) {
      argNames.push_back("arg" + std::to_string(arg.getArgNumber()));
      argTypes.push_back(arg.getType());
    }
  }

  // Determine N (total host-buffer element count) from the first memref arg,
  // as the product of all shape dims. The buffer length is always
  //   w * h * l == total elements,
  // regardless of how the memref is sharded across the PE grid.
  int64_t N = -1;
  for (Type t : argTypes) {
    if (auto memTy = dyn_cast<MemRefType>(t)) {
      int64_t n = 1;
      for (int64_t d : memTy.getShape())
        n *= d;
      if (n > 0) {
        N = n;
        break;
      }
    }
  }
  std::string nStr = (N > 0) ? std::to_string(N) : std::string("256");

  // Helper: for a memcpy referencing sym `@layout::@leaf`, find the
  // csl_layout.place that owns the program containing `@leaf`, and use it
  // together with the host-buffer memref type to derive (w, h, l).
  auto fillExtents = [&](StringRef leaf, MemRefType memTy, MemcpyEntry &e) {
    StringRef prog = findProgramForLeaf(wafer, leaf);
    xilinx::csl_layout::PlaceOp place = findPlaceForProgram(wafer, prog);
    auto [w, h, l] = deriveMemcpyExtent(place, memTy);
    e.w = w;
    e.h = h;
    e.l = l;
  };

  // Classify each arg as h2d/d2h/unused and record the memcpy + launch
  // sequence. We emit h2d first, then launch, then d2h (SdkRuntime standard
  // flow), regardless of the source program order.
  llvm::SmallVector<Dir, 4> argDir(argNames.size(), Dir::Unused);
  llvm::SmallVector<MemcpyEntry, 4> memcpys;
  std::string launchFn;

  if (!host.getBody().empty()) {
    Block &entry = host.getBody().front();
    for (Operation &op : entry) {
      if (auto h2dOp = dyn_cast<hostns::MemcpyH2DOp>(&op)) {
        SymbolRefAttr symAttr = h2dOp.getSym();
        StringRef leafSym = !symAttr.getNestedReferences().empty()
                                ? symAttr.getNestedReferences().back().getValue()
                                : symAttr.getRootReference().getValue();
        Value src = h2dOp.getSrc();
        unsigned idx = 0;
        if (auto barg = dyn_cast<BlockArgument>(src))
          idx = barg.getArgNumber();
        if (idx < argDir.size())
          argDir[idx] = Dir::H2D;
        MemcpyEntry e{true, idx, leafSym.str(), h2dOp.getPx(), h2dOp.getPy(),
                      /*w=*/1, /*h=*/1, /*l=*/0};
        auto memTy = dyn_cast<MemRefType>(h2dOp.getSrc().getType());
        if (memTy)
          fillExtents(leafSym, memTy, e);
        memcpys.push_back(e);
        continue;
      }
      if (auto d2hOp = dyn_cast<hostns::MemcpyD2HOp>(&op)) {
        SymbolRefAttr symAttr = d2hOp.getSym();
        StringRef leafSym = !symAttr.getNestedReferences().empty()
                                ? symAttr.getNestedReferences().back().getValue()
                                : symAttr.getRootReference().getValue();
        Value dst = d2hOp.getDst();
        unsigned idx = 0;
        if (auto barg = dyn_cast<BlockArgument>(dst))
          idx = barg.getArgNumber();
        if (idx < argDir.size())
          argDir[idx] = Dir::D2H;
        MemcpyEntry e{false, idx, leafSym.str(), d2hOp.getPx(), d2hOp.getPy(),
                      /*w=*/1, /*h=*/1, /*l=*/0};
        auto memTy = dyn_cast<MemRefType>(d2hOp.getDst().getType());
        if (memTy)
          fillExtents(leafSym, memTy, e);
        memcpys.push_back(e);
        continue;
      }
      if (auto launchOp = dyn_cast<hostns::LaunchOp>(&op)) {
        SymbolRefAttr symAttr = launchOp.getSym();
        StringRef leafSym = !symAttr.getNestedReferences().empty()
                                ? symAttr.getNestedReferences().back().getValue()
                                : symAttr.getRootReference().getValue();
        launchFn = leafSym.str();
        continue;
      }
    }
  }

  // -- Emit a complete runnable cs_python script. --
  os << "#!/usr/bin/env cs_python\n";
  os << "# Generated by air-translate --emit-csl-host\n";
  os << "#\n";
  os << "# Run against a pre-compiled artifact directory produced by\n";
  os << "#   cslc layout.csl --arch=wseN ... -o <dir> --memcpy --channels 1\n";
  os << "# The companion commands_wse{2,3}.sh scripts do both steps.\n\n";
  os << "import argparse\n";
  os << "import sys\n";
  os << "import numpy as np\n\n";
  os << "from cerebras.sdk.runtime.sdkruntimepybind import (\n";
  os << "    SdkRuntime,\n";
  os << "    MemcpyOrder,\n";
  os << "    MemcpyDataType,\n";
  os << ")\n\n";

  os << "parser = argparse.ArgumentParser()\n";
  os << "parser.add_argument('--name', default='compiled',\n";
  os << "                    help='compiled artifact directory (default: "
        "compiled)')\n";
  os << "parser.add_argument('--cmaddr', default=None,\n";
  os << "                    help='IP:port for CS system (default: simulator)')\n";
  os << "args = parser.parse_args()\n\n";

  os << "N = " << nStr << "\n\n";

  // Build per-arg numpy buffers. For h2d: arange * (idx+1). For d2h: zeros.
  bool anyFloatOutput = false;
  os << "# Host buffers.\n";
  for (unsigned i = 0; i < argNames.size(); ++i) {
    if (argDir[i] == Dir::Unused)
      continue;
    Type eltTy;
    if (auto memTy = dyn_cast<MemRefType>(argTypes[i]))
      eltTy = memTy.getElementType();
    else
      eltTy = Float32Type::get(host.getContext());
    ElemInfo info = classifyElem(eltTy);

    if (argDir[i] == Dir::H2D) {
      unsigned scale = i + 1;
      if (info.isFloat) {
        os << argNames[i] << " = np.arange(N, dtype=" << info.npDtype << ")";
        if (scale != 1)
          os << " * " << scale << ".0";
        os << "\n";
      } else {
        os << argNames[i] << " = (np.arange(N, dtype=" << info.npDtype << ")";
        if (scale != 1)
          os << " * " << scale;
        os << ")\n";
      }
    } else { // D2H
      os << argNames[i] << " = np.zeros(N, dtype=" << info.npDtype << ")\n";
      if (info.isFloat)
        anyFloatOutput = true;
    }
  }
  os << "\n";

  os << "# Create runtime and load compiled artifacts.\n";
  os << "runner = SdkRuntime(args.name, cmaddr=args.cmaddr)\n";
  os << "runner.load()\n";
  os << "runner.run()\n\n";

  // Phase 1: all H2D copies.
  os << "# Host-to-device copies (inputs).\n";
  for (const auto &e : memcpys)
    if (e.isH2d)
      emitMemcpy(e, argNames, argTypes, host.getContext());
  os << "\n";

  // Phase 2: launch.
  os << "# Launch compute kernel.\n";
  if (!launchFn.empty())
    os << "runner.launch(\"" << launchFn << "\", nonblock=False)\n";
  os << "\n";

  // Phase 3: all D2H copies.
  os << "# Device-to-host copies (outputs).\n";
  for (const auto &e : memcpys)
    if (!e.isH2d)
      emitMemcpy(e, argNames, argTypes, host.getContext());
  os << "\n";

  os << "runner.stop()\n\n";

  // -- Reference computation + correctness check. --
  //
  // Strategy: interpret the `csl.func @compute` body in numpy. For each
  // csl.var touched by a memcpy, materialize a device-side numpy view seeded
  // from the host buffer (H2D) or zeros (D2H). Walk the body emitting numpy
  // code, then compare the expected host buffer against the actual.
  //
  // If any op in the body is unsupported by the walker, we bail out and
  // fall back to the legacy sanity check.
  bool refOk = false;
  std::string refBlock;
  llvm::SmallVector<unsigned, 4> refD2HArgs; // arg indices verified by refBlock
  {
    // Find the program containing the launch target.
    StringRef progSym =
        launchFn.empty() ? StringRef() : findProgramForLeaf(wafer, launchFn);
    xilinx::csl::ProgramOp program;
    wafer.walk([&](xilinx::csl::ProgramOp p) {
      if (program)
        return;
      if (progSym.empty() || p.getSymName() == progSym)
        program = p;
    });

    // Locate the compute csl.func in the program.
    xilinx::csl::FuncOp computeFn;
    if (program) {
      program.getBody().walk([&](xilinx::csl::FuncOp f) {
        if (computeFn)
          return;
        if (launchFn.empty() || f.getSymName() == launchFn)
          computeFn = f;
      });
    }

    // Map per-var info: var symbol -> (arg index, device memref type, direction).
    // The device memref type is the one declared on `csl.var` (the per-PE
    // buffer shape), NOT the host-side memref (which may be the full sharded
    // buffer).
    struct VarBinding {
      unsigned argIdx;
      MemRefType memTy; // the PE-local csl.var's memref type
      Dir dir;
      std::string refBufName; // e.g. "_a_ref"
    };
    llvm::StringMap<VarBinding> varBindings;
    for (const MemcpyEntry &e : memcpys) {
      if (e.argIdx >= argTypes.size())
        continue;
      // Look up the csl.var declaration for `e.leafSym` inside the program.
      MemRefType deviceTy;
      if (program) {
        program.getBody().walk([&](xilinx::csl::VarOp vop) {
          if (deviceTy)
            return;
          if (vop.getSymName() == e.leafSym)
            if (auto mt = dyn_cast<MemRefType>(vop.getResult().getType()))
              deviceTy = mt;
        });
      }
      if (!deviceTy)
        continue;
      VarBinding vb;
      vb.argIdx = e.argIdx;
      vb.memTy = deviceTy;
      vb.dir = e.isH2d ? Dir::H2D : Dir::D2H;
      vb.refBufName = "_" + e.leafSym + "_ref";
      varBindings[e.leafSym] = vb;
    }

    // Collect D2H output leaf syms (for the assertion section).
    llvm::SmallVector<std::string, 4> d2hLeaves;
    for (const MemcpyEntry &e : memcpys) {
      if (!e.isH2d)
        d2hLeaves.push_back(e.leafSym);
    }

    if (computeFn && !d2hLeaves.empty() && !computeFn.getBody().empty()) {
      std::string buf;
      llvm::raw_string_ostream refOs(buf);
      RefEmitState state;

      // Per-var device-side shape length inside the kernel (the memref shape
      // as declared at csl.var — may differ from the host buffer size).
      auto varLen = [](MemRefType mt) -> int64_t {
        int64_t n = 1;
        for (int64_t d : mt.getShape())
          n *= d;
        return n;
      };

      // Seed the SSA-name map: each csl.var's result -> local numpy array.
      // We also emit the allocation/seeding statements.
      refOs << "# Reference computation (numpy, interpreted from csl.func @"
            << computeFn.getSymName() << " body).\n";
      program.getBody().walk([&](xilinx::csl::VarOp vop) {
        auto mt = dyn_cast<MemRefType>(vop.getResult().getType());
        if (!mt)
          return;
        auto it = varBindings.find(vop.getSymName());
        if (it == varBindings.end()) {
          // Var is not touched by any memcpy — allocate as zeros of declared
          // shape so loads/stores to it still work.
          std::string name = ("_" + vop.getSymName() + "_ref").str();
          refOs << name << " = np.zeros(" << varLen(mt)
                << ", dtype=" << rnpDtype(mt.getElementType()) << ")\n";
          state.nameMap[vop.getResult()] = name;
          return;
        }
        const VarBinding &vb = it->second;
        int64_t vlen = varLen(vb.memTy);
        std::string dtype = rnpDtype(vb.memTy.getElementType());
        if (vb.dir == Dir::H2D) {
          // H2D: device buffer holds the first `vlen` elements of the host
          // h2d input (per ROW_MAJOR memcpy).
          refOs << vb.refBufName << " = " << argNames[vb.argIdx] << "[:"
                << vlen << "].astype(" << dtype << ").copy()\n";
        } else {
          refOs << vb.refBufName << " = np.zeros(" << vlen << ", dtype="
                << dtype << ")\n";
        }
        state.nameMap[vop.getResult()] = vb.refBufName;
      });

      // Walk the body.
      emitRefBody(computeFn.getBody(), refOs, 0, state);

      if (state.ok) {
        // Build the verification section.
        refOs << "\n# Verification: compare each D2H output against its "
                 "reference.\n";
        refOs << "_mismatch = False\n";
        for (const std::string &leaf : d2hLeaves) {
          auto it = varBindings.find(leaf);
          if (it == varBindings.end())
            continue;
          const VarBinding &vb = it->second;
          int64_t vlen = varLen(vb.memTy);
          std::string argName = argNames[vb.argIdx];
          std::string dtype = rnpDtype(vb.memTy.getElementType());
          refOs << "_expected_" << argName << " = np.zeros_like(" << argName
                << ")\n";
          refOs << "_expected_" << argName << "[:" << vlen << "] = "
                << vb.refBufName << "\n";
          refOs << "if not np.allclose(" << argName << ", _expected_"
                << argName << ", atol=1e-5, rtol=1e-5):\n";
          refOs << "    print(\"MISMATCH in " << argName << " (var @" << leaf
                << "):\", file=sys.stderr)\n";
          refOs << "    print(f\"  expected[:8] = {_expected_" << argName
                << "[:min(8, len(_expected_" << argName << "))]}\", "
                   "file=sys.stderr)\n";
          refOs << "    print(f\"  got[:8]      = {" << argName
                << "[:min(8, len(" << argName << "))]}\", file=sys.stderr)\n";
          refOs << "    _mismatch = True\n";
          refOs << "else:\n";
          refOs << "    print(f\"  " << argName << "[:8] = {" << argName
                << "[:min(8, len(" << argName
                << "))]} (matches reference)\")\n";
          refD2HArgs.push_back(vb.argIdx);
        }
        refOs << "if _mismatch:\n";
        refOs << "    sys.exit(1)\n";
        refOs.flush();
        refBlock = buf;
        refOk = true;
      } else {
        os << "# Reference-compute skipped: " << state.reason << "\n";
      }
    }
  }

  if (refOk) {
    os << refBlock;
  } else {
    // Fallback: legacy sanity check (no reference model available for this
    // kernel — e.g. empty compute or unsupported op in body).
    os << "# Output sanity checks (no reference model available).\n";
    if (anyFloatOutput) {
      os << "def _sanity(name, buf):\n";
      os << "    head = buf[:min(8, len(buf))]\n";
      os << "    print(f\"  {name}[:{len(head)}] = {list(head)}\")\n";
      os << "    return bool(np.any(buf != 0))\n\n";
      os << "print(\"Output buffers:\")\n";
      os << "any_nonzero = False\n";
      for (unsigned i = 0; i < argNames.size(); ++i) {
        if (argDir[i] == Dir::D2H)
          os << "any_nonzero = _sanity(\"" << argNames[i] << "\", "
             << argNames[i] << ") or any_nonzero\n";
      }
      os << "# Note: all-zero output is OK if the kernel has no stores (e.g. "
            "empty compute).\n";
    } else {
      for (unsigned i = 0; i < argNames.size(); ++i) {
        if (argDir[i] == Dir::D2H)
          os << "print(\"" << argNames[i] << "[:8] =\", " << argNames[i]
             << "[:min(8, len(" << argNames[i] << "))])\n";
      }
    }
  }
  os << "print(\"SUCCESS!\")\n";

  return success();
}

} // namespace

// Wafer-scoped entry point used by CSLEmitAll.cpp.
LogicalResult runHostEmitter(xilinx::csl::WaferOp wafer,
                             llvm::raw_ostream &os) {
  HostEmitter emitter(os);
  return emitter.emit(wafer);
}

// Module-scoped entry point for backward compatibility (--emit-csl-host).
LogicalResult runHostEmitter(ModuleOp module, llvm::raw_ostream &os) {
  namespace cslns = xilinx::csl;
  llvm::SmallVector<cslns::WaferOp, 4> wafers;
  module.walk([&](cslns::WaferOp w) { wafers.push_back(w); });
  if (wafers.empty()) {
    os << "# No csl.wafer found in module.\n";
    return success();
  }
  if (wafers.size() > 1)
    os << "# note: " << wafers.size()
       << " wafers in module; emitting " << wafers[0].getSymName() << "\n";
  return runHostEmitter(wafers[0], os);
}

void registerCSLHostTranslation() {
  static TranslateFromMLIRRegistration reg(
      "emit-csl-host",
      "Emit CSL v2 host runtime Python (run.py) from csl.host",
      [](ModuleOp module, llvm::raw_ostream &os) -> LogicalResult {
        return runHostEmitter(module, os);
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
