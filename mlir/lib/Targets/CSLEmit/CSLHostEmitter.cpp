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

  // Output sanity check: skip reference verification (the emitter has no
  // reference model). Just dump first elements and warn if outputs are all
  // zero (implying the kernel didn't run).
  os << "# Output sanity checks (no reference model is known to the emitter).\n";
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
    os << "if not any_nonzero:\n";
    os << "    print(\"WARNING: all output buffers are zero - kernel may not "
          "have run.\", file=sys.stderr)\n";
  } else {
    for (unsigned i = 0; i < argNames.size(); ++i) {
      if (argDir[i] == Dir::D2H)
        os << "print(\"" << argNames[i] << "[:8] =\", " << argNames[i]
           << "[:min(8, len(" << argNames[i] << "))])\n";
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
