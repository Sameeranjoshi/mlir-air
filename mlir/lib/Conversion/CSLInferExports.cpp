//===- CSLInferExports.cpp - Auto-generate csl.export ops -------*- C++ -*-===//
//
// Part of the air-to-csl project.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
//
// Implements the -csl-infer-exports pass.
//
// Algorithm (per csl.wafer):
//   1. Walk csl.host regions and collect {sym_leaf → direction/kind} from:
//      - csl_host.memcpy_h2d → direction = "in"
//      - csl_host.memcpy_d2h → direction = "out"
//      - csl_host.launch     → kind = "func"
//   2. For each collected symbol, if no csl.export exists in the program,
//      create one.  If no csl_layout.export exists in the layout, create one.
//   3. Pre-existing csl.export ops without a direction receive "internal".
//
//===----------------------------------------------------------------------===//

#include "air/Conversion/CSLInferExportsPass.h"
#include "air/Dialect/CSL/CSLHostOps.h"
#include "air/Dialect/CSL/CSLLayoutOps.h"
#include "air/Dialect/CSL/CSLOps.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSet.h"

using namespace mlir;

namespace {

/// Returns the leaf reference name of a SymbolRefAttr (the last nested ref,
/// or the root if there are no nested refs).
static StringRef leafRef(SymbolRefAttr sym) {
  if (sym.getNestedReferences().empty())
    return sym.getRootReference().getValue();
  return sym.getNestedReferences().back().getRootReference().getValue();
}

/// Information collected about one symbol from host ops.
struct SymInfo {
  std::string direction; // "in", "out", or empty
  bool isFunc = false;   // true if referenced by csl_host.launch
};

/// Collected host symbols: a map for lookup plus a vector for insertion order.
struct HostSymbols {
  llvm::StringMap<SymInfo> map;
  SmallVector<std::string> order; // insertion order (unique keys)

  SymInfo &getOrInsert(StringRef key) {
    auto it = map.find(key);
    if (it != map.end())
      return it->second;
    order.push_back(key.str());
    return map[key];
  }
};

/// Walk csl.host regions and collect export info keyed by sym leaf name.
static HostSymbols collectHostSymbols(xilinx::csl::WaferOp wafer) {
  HostSymbols result;

  wafer.getBody().walk([&](xilinx::csl::HostOp host) {
    host.getBody().walk([&](Operation *op) {
      if (auto h2d = dyn_cast<xilinx::csl_host::MemcpyH2DOp>(op)) {
        StringRef leaf = leafRef(h2d.getSym());
        auto &info = result.getOrInsert(leaf);
        if (info.direction.empty())
          info.direction = "in";
      } else if (auto d2h = dyn_cast<xilinx::csl_host::MemcpyD2HOp>(op)) {
        StringRef leaf = leafRef(d2h.getSym());
        auto &info = result.getOrInsert(leaf);
        if (info.direction.empty())
          info.direction = "out";
      } else if (auto launch = dyn_cast<xilinx::csl_host::LaunchOp>(op)) {
        StringRef leaf = leafRef(launch.getSym());
        auto &info = result.getOrInsert(leaf);
        info.isFunc = true;
      }
    });
  });

  return result;
}

/// Find the first csl.program child of a wafer.
static xilinx::csl::ProgramOp findProgram(xilinx::csl::WaferOp wafer) {
  xilinx::csl::ProgramOp prog;
  wafer.getBody().walk([&](xilinx::csl::ProgramOp p) {
    if (!prog)
      prog = p;
  });
  return prog;
}

/// Find the first csl.layout child of a wafer.
static xilinx::csl::LayoutOp findLayout(xilinx::csl::WaferOp wafer) {
  xilinx::csl::LayoutOp layout;
  wafer.getBody().walk([&](xilinx::csl::LayoutOp l) {
    if (!layout)
      layout = l;
  });
  return layout;
}

/// Collect existing csl.export ops in a program, keyed by sym name.
static llvm::StringMap<xilinx::csl::ExportOp>
collectExistingExports(xilinx::csl::ProgramOp prog) {
  llvm::StringMap<xilinx::csl::ExportOp> result;
  prog.getBody().walk([&](xilinx::csl::ExportOp exp) {
    result[exp.getSym()] = exp;
  });
  return result;
}

/// Collect existing csl_layout.export sym_names in a layout.
static llvm::StringSet<>
collectExistingLayoutExports(xilinx::csl::LayoutOp layout) {
  llvm::StringSet<> result;
  layout.getBody().walk([&](xilinx::csl_layout::ExportOp exp) {
    result.insert(exp.getSymName());
  });
  return result;
}

/// Main pass logic for one wafer.
static void inferExports(xilinx::csl::WaferOp wafer) {
  MLIRContext *ctx = wafer.getContext();

  auto prog = findProgram(wafer);
  auto layout = findLayout(wafer);
  if (!prog)
    return;

  // Collect host-referenced symbols.
  HostSymbols hostSyms = collectHostSymbols(wafer);

  // Collect existing exports.
  auto existingProgramExports = collectExistingExports(prog);
  llvm::StringSet<> existingLayoutExports;
  if (layout)
    existingLayoutExports = collectExistingLayoutExports(layout);

  // Get program name for layout export "from" references.
  StringRef progName = prog.getSymName();

  Location loc = wafer.getLoc();

  // Insert new csl.export ops or update existing ones with host-derived info.
  {
    OpBuilder pb(ctx);
    pb.setInsertionPointToEnd(&prog.getBody().front());

    for (const auto &symLeaf : hostSyms.order) {
      const SymInfo &info = hostSyms.map[symLeaf];

      auto it = existingProgramExports.find(symLeaf);
      if (it != existingProgramExports.end()) {
        // Export already exists — update its direction/kind from host info.
        xilinx::csl::ExportOp existing = it->second;
        if (info.isFunc && !existing.getKind())
          existing->setAttr("kind", StringAttr::get(ctx, "func"));
        if (!info.isFunc && !info.direction.empty() && !existing.getDirection())
          existing->setAttr("direction",
                            StringAttr::get(ctx, info.direction));
        continue;
      }

      if (info.isFunc) {
        // Function export: no alias, kind = "func", direction will be set
        // below (it won't have a direction from host memcpy, so "internal").
        xilinx::csl::ExportOp::create(
            pb, loc, FlatSymbolRefAttr::get(ctx, symLeaf),
            /*alias=*/StringAttr{},
            /*kind=*/pb.getStringAttr("func"),
            /*direction=*/StringAttr{});
      } else {
        // Variable export: alias = symLeaf, direction from host ops.
        xilinx::csl::ExportOp::create(
            pb, loc, FlatSymbolRefAttr::get(ctx, symLeaf),
            /*alias=*/pb.getStringAttr(symLeaf),
            /*kind=*/StringAttr{},
            /*direction=*/pb.getStringAttr(info.direction));
      }
    }
  }

  // Insert new csl_layout.export ops at the end of the layout body.
  if (layout) {
    OpBuilder lb(ctx);
    lb.setInsertionPointToEnd(&layout.getBody().front());

    auto progSymAttr = StringAttr::get(ctx, progName);

    for (const auto &symLeaf : hostSyms.order) {
      if (existingLayoutExports.contains(symLeaf))
        continue;

      const SymInfo &info = hostSyms.map[symLeaf];

      SymbolRefAttr fromRef = SymbolRefAttr::get(
          progSymAttr, {FlatSymbolRefAttr::get(ctx, symLeaf)});

      if (info.isFunc) {
        xilinx::csl_layout::ExportOp::create(
            lb, loc, lb.getStringAttr(symLeaf), fromRef,
            lb.getStringAttr("func"));
      } else {
        xilinx::csl_layout::ExportOp::create(
            lb, loc, lb.getStringAttr(symLeaf), fromRef,
            /*kind=*/StringAttr{});
      }
    }
  }

  // Any csl.export without a direction receives "internal".
  prog.getBody().walk([&](xilinx::csl::ExportOp exp) {
    if (!exp.getDirection())
      exp->setAttr("direction", StringAttr::get(ctx, "internal"));
  });
}

//===----------------------------------------------------------------------===//
// Pass definition
//===----------------------------------------------------------------------===//

struct CSLInferExportsPass
    : public PassWrapper<CSLInferExportsPass, OperationPass<ModuleOp>> {

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(CSLInferExportsPass)

  StringRef getArgument() const override { return "csl-infer-exports"; }
  StringRef getDescription() const override {
    return "Auto-generate csl.export and csl_layout.export ops from "
           "csl_host transfer and launch ops";
  }

  void runOnOperation() override {
    ModuleOp mod = getOperation();
    mod.walk([&](xilinx::csl::WaferOp wafer) { inferExports(wafer); });
  }
};

} // anonymous namespace

std::unique_ptr<OperationPass<ModuleOp>>
xilinx::air::createCSLInferExportsPass() {
  return std::make_unique<CSLInferExportsPass>();
}
