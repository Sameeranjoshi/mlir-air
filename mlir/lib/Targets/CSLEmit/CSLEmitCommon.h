//===- CSLEmitCommon.h - Shared helpers for CSL emitters -------*- C++ -*-===//
//
// Part of the air-to-csl project.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
//
// Small shared helpers used by the CSL program/layout/host emitters.
// Defined as `static inline` so each translation unit gets its own copy;
// this keeps the CSLEmit/ directory free of a separate common.cpp.
//
//===----------------------------------------------------------------------===//

#ifndef AIR_TARGETS_CSLEMIT_COMMON_H
#define AIR_TARGETS_CSLEMIT_COMMON_H

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
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/raw_ostream.h"

#include <string>

namespace xilinx {
namespace csl {
namespace detail {

/// Return the CSL primitive type name for an MLIR type.
static inline std::string cslTypeName(mlir::Type t) {
  if (t.isF32()) return "f32";
  if (t.isF16()) return "f16";
  if (t.isInteger(32)) return "i32";
  if (t.isInteger(16)) return "i16";
  if (t.isIndex()) return "u16";
  return "f32"; // fallback
}

/// Write N spaces of indentation (2 spaces per level).
static inline void indent(llvm::raw_ostream &os, unsigned level) {
  for (unsigned i = 0; i < level; ++i)
    os << "  ";
}

/// Resolve a value's string representation from nameMap, or return "?".
static inline std::string
resolve(const llvm::DenseMap<mlir::Value, std::string> &nameMap,
        mlir::Value v) {
  auto it = nameMap.find(v);
  if (it != nameMap.end())
    return it->second;
  return "?";
}

/// Emit CSL function body ops into `os`. `outerMap` maps values defined
/// *outside* the function (e.g. csl.var results) to their names.
static inline mlir::LogicalResult
emitFuncBody(mlir::Region &bodyRegion, llvm::raw_ostream &os,
             unsigned indentLevel,
             const llvm::DenseMap<mlir::Value, std::string> &outerMap,
             llvm::DenseMap<mlir::Value, std::string> &nameMap,
             unsigned &tempCount) {
  using namespace mlir;

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
      op.emitOpError("CSLEmit: unsupported op in function body: ");
      return failure();
    }
  }
  return mlir::success();
}

} // namespace detail
} // namespace csl
} // namespace xilinx

#endif // AIR_TARGETS_CSLEMIT_COMMON_H
