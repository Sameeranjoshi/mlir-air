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
///
/// `index` → `u16`: matches the SDK tutorials' canonical loop-counter form
/// (Builtins.md:1169 `for(@range(u16, 8))`, gemv-02). DSD `extent` is also
/// `u16` (DSDs.md:36), so unsigned indexing flows naturally into DSD shape.
static inline std::string cslTypeName(mlir::Type t) {
  if (t.isF32()) return "f32";
  if (t.isF16()) return "f16";
  if (t.isInteger(1))  return "bool";
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

/// True if `s` is a pure numeric literal (e.g. "42", "-1.5", "2.000000",
/// "1.5e-3"). Used to spot arith.constant-sourced values that CSL treats as
/// `comptime_float` / `comptime_int` — those can't flow as runtime args to
/// module-member calls like `math.sqrt(...)` without a typed cast.
static inline bool isNumericLiteral(llvm::StringRef s) {
  if (s.empty()) return false;
  size_t i = 0;
  if (s[i] == '-' || s[i] == '+') ++i;
  bool hasDigit = false, hasDot = false, hasExp = false;
  for (; i < s.size(); ++i) {
    char c = s[i];
    if (c >= '0' && c <= '9') hasDigit = true;
    else if (c == '.' && !hasDot && !hasExp) hasDot = true;
    else if ((c == 'e' || c == 'E') && !hasExp && hasDigit) hasExp = true;
    else if ((c == '+' || c == '-') && hasExp
             && (s[i-1] == 'e' || s[i-1] == 'E')) { /* exponent sign */ }
    else return false;
  }
  return hasDigit;
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

      // arith.addf / arith.addi — type comes from the result; addi on index
      // values emits u16, on i32 emits i32, etc.
      if (auto addOp = dyn_cast<arith::AddFOp>(&op)) {
        std::string lhs = resolve(nameMap, addOp.getLhs());
        std::string rhs = resolve(nameMap, addOp.getRhs());
        std::string tname = "t" + std::to_string(tempCount++);
        indent(os, indentLevel);
        os << "var " << tname << ": "
           << cslTypeName(addOp.getResult().getType()) << " = "
           << lhs << " + " << rhs << ";\n";
        nameMap[addOp.getResult()] = tname;
        continue;
      }
      if (auto addOp = dyn_cast<arith::AddIOp>(&op)) {
        std::string lhs = resolve(nameMap, addOp.getLhs());
        std::string rhs = resolve(nameMap, addOp.getRhs());
        std::string tname = "t" + std::to_string(tempCount++);
        indent(os, indentLevel);
        os << "var " << tname << ": "
           << cslTypeName(addOp.getResult().getType()) << " = "
           << lhs << " + " << rhs << ";\n";
        nameMap[addOp.getResult()] = tname;
        continue;
      }

      // Helper lambdas for new binary arith ops.
      // Emit: `var tN: <csl-ty> = lhs OP rhs;`
      auto emitBinary = [&](Operation *bop, StringRef opSym) {
        std::string l = resolve(nameMap, bop->getOperand(0));
        std::string r = resolve(nameMap, bop->getOperand(1));
        std::string tname = "t" + std::to_string(tempCount++);
        indent(os, indentLevel);
        os << "var " << tname << ": "
           << cslTypeName(bop->getResult(0).getType()) << " = " << l << " "
           << opSym << " " << r << ";\n";
        nameMap[bop->getResult(0)] = tname;
      };
      // Emit a scalar min/max as a CSL if-expression:
      //   var tN: <csl-ty> = if (lhs CMP rhs) lhs else rhs;
      // CSL has no scalar @max/@min builtin (@fmaxs/@fmaxh operate on DSDs).
      auto emitMinMax = [&](Operation *bop, StringRef cmp) {
        std::string l = resolve(nameMap, bop->getOperand(0));
        std::string r = resolve(nameMap, bop->getOperand(1));
        std::string tname = "t" + std::to_string(tempCount++);
        indent(os, indentLevel);
        os << "var " << tname << ": "
           << cslTypeName(bop->getResult(0).getType()) << " = if (" << l
           << " " << cmp << " " << r << ") " << l << " else " << r << ";\n";
        nameMap[bop->getResult(0)] = tname;
      };

      // Float binary ops
      if (dyn_cast<arith::SubFOp>(&op)) { emitBinary(&op, "-"); continue; }
      if (dyn_cast<arith::MulFOp>(&op)) { emitBinary(&op, "*"); continue; }
      if (dyn_cast<arith::DivFOp>(&op)) { emitBinary(&op, "/"); continue; }
      if (dyn_cast<arith::MaximumFOp>(&op)) { emitMinMax(&op, ">"); continue; }
      if (dyn_cast<arith::MinimumFOp>(&op)) { emitMinMax(&op, "<"); continue; }

      // Integer binary ops
      if (dyn_cast<arith::SubIOp>(&op)) { emitBinary(&op, "-"); continue; }
      if (dyn_cast<arith::MulIOp>(&op)) { emitBinary(&op, "*"); continue; }
      if (dyn_cast<arith::RemSIOp>(&op)) { emitBinary(&op, "%"); continue; }
      if (dyn_cast<arith::RemUIOp>(&op)) { emitBinary(&op, "%"); continue; }
      if (dyn_cast<arith::DivSIOp>(&op)) { emitBinary(&op, "/"); continue; }
      if (dyn_cast<arith::DivUIOp>(&op)) { emitBinary(&op, "/"); continue; }

      // Bitwise (wider integers) or logical (i1) ops. CSL separates the two:
      // bool uses `or`/`and` (Syntax.md operator table); integer uses `|`/`&`/`^`.
      // CSL has no logical-xor keyword on bool — `a != b` is the canonical form.
      auto orWord = [](Operation *bop) {
        return bop->getResult(0).getType().isInteger(1) ? "or"  : "|";
      };
      auto andWord = [](Operation *bop) {
        return bop->getResult(0).getType().isInteger(1) ? "and" : "&";
      };
      auto xorWord = [](Operation *bop) {
        return bop->getResult(0).getType().isInteger(1) ? "!=" : "^";
      };
      if (dyn_cast<arith::OrIOp>(&op))  { emitBinary(&op, orWord(&op));  continue; }
      if (dyn_cast<arith::AndIOp>(&op)) { emitBinary(&op, andWord(&op)); continue; }
      if (dyn_cast<arith::XOrIOp>(&op)) { emitBinary(&op, xorWord(&op)); continue; }

      // arith.cmpf — translate predicate → CSL operator; reject unordered forms.
      if (auto cmp = dyn_cast<arith::CmpFOp>(&op)) {
        using P = arith::CmpFPredicate;
        const char *sym = nullptr;
        switch (cmp.getPredicate()) {
        case P::OEQ: sym = "=="; break;
        case P::OLT: sym = "<";  break;
        case P::OLE: sym = "<="; break;
        case P::OGT: sym = ">";  break;
        case P::OGE: sym = ">="; break;
        case P::ONE: sym = "!="; break;
        default:
          op.emitError("arith.cmpf unordered predicates unsupported in v5");
          return failure();
        }
        std::string l = resolve(nameMap, cmp.getLhs());
        std::string r = resolve(nameMap, cmp.getRhs());
        std::string tname = "b" + std::to_string(tempCount++);
        indent(os, indentLevel);
        os << "const " << tname << " = " << l << " " << sym << " " << r
           << ";\n";
        nameMap[cmp.getResult()] = tname;
        continue;
      }

      // arith.cmpi — same mapping as cmpf; signed and unsigned collapse.
      if (auto cmp = dyn_cast<arith::CmpIOp>(&op)) {
        using P = arith::CmpIPredicate;
        const char *sym = nullptr;
        switch (cmp.getPredicate()) {
        case P::eq:  sym = "==";  break;
        case P::ne:  sym = "!=";  break;
        case P::slt: case P::ult: sym = "<";  break;
        case P::sle: case P::ule: sym = "<="; break;
        case P::sgt: case P::ugt: sym = ">";  break;
        case P::sge: case P::uge: sym = ">="; break;
        }
        std::string l = resolve(nameMap, cmp.getLhs());
        std::string r = resolve(nameMap, cmp.getRhs());
        std::string tname = "b" + std::to_string(tempCount++);
        indent(os, indentLevel);
        os << "const " << tname << " = " << l << " " << sym << " " << r
           << ";\n";
        nameMap[cmp.getResult()] = tname;
        continue;
      }

      // arith.index_cast / arith.index_castui — emit @as(<T>, v)
      if (auto cast = dyn_cast<arith::IndexCastOp>(&op)) {
        std::string a = resolve(nameMap, cast.getIn());
        std::string tname = "t" + std::to_string(tempCount++);
        std::string ty = cslTypeName(cast.getResult().getType());
        indent(os, indentLevel);
        os << "const " << tname << ": " << ty << " = @as(" << ty << ", " << a
           << ");\n";
        nameMap[cast.getResult()] = tname;
        continue;
      }

      // scf.if (no yielded values).
      if (auto ifOp = dyn_cast<scf::IfOp>(&op)) {
        if (ifOp.getNumResults() != 0) {
          op.emitError("scf.if with yielded values is unsupported in v5");
          return failure();
        }
        std::string cond = resolve(nameMap, ifOp.getCondition());
        indent(os, indentLevel);
        os << "if (" << cond << ") {\n";
        if (failed(emitFuncBody(ifOp.getThenRegion(), os, indentLevel + 1,
                                outerMap, nameMap, tempCount)))
          return failure();
        indent(os, indentLevel);
        os << "}";
        if (!ifOp.getElseRegion().empty() &&
            !ifOp.getElseRegion().front().empty()) {
          os << " else {\n";
          if (failed(emitFuncBody(ifOp.getElseRegion(), os, indentLevel + 1,
                                  outerMap, nameMap, tempCount)))
            return failure();
          indent(os, indentLevel);
          os << "}";
        }
        os << "\n";
        continue;
      }

      // scf.index_switch → CSL switch (@as(i32, arg)) { N => { body }, ..., else => { default } }
      // The CSL switch scrutinee must be a fixed-width integer, so we cast the
      // index-typed arg through @as(i32, ...). Case labels are integer literals
      // which CSL accepts against an i32 scrutinee. The default region is
      // always present in scf.index_switch (MLIR spec).
      if (auto sw = dyn_cast<scf::IndexSwitchOp>(&op)) {
        if (sw.getNumResults() != 0) {
          op.emitError("scf.index_switch with yielded values is unsupported in v5");
          return failure();
        }
        std::string arg = resolve(nameMap, sw.getArg());
        indent(os, indentLevel);
        // CSL switch requires a fixed-width integer scrutinee. The argument is
        // index-typed, which maps to u16 in CSL — already a valid fixed-width
        // integer. Case literals are comptime_int and are coercible to u16
        // (Syntax.md: "case_value expressions must be comptime-known and
        // coercible to the type of the input expression"). Do NOT add an
        // @as(i32, ...) wrapper: the SDK rejects extsi u16->i32 as a
        // "cast incompatible" error on its internal lowering path.
        os << "switch (" << arg << ") {\n";
        // Emit each case region alongside its label.
        auto caseVals = sw.getCases();
        auto caseRegions = sw.getCaseRegions();
        for (unsigned idx = 0; idx < caseVals.size(); ++idx) {
          indent(os, indentLevel + 1);
          os << caseVals[idx] << " => {\n";
          if (failed(emitFuncBody(caseRegions[idx], os, indentLevel + 2,
                                  outerMap, nameMap, tempCount)))
            return failure();
          indent(os, indentLevel + 1);
          os << "},\n";
        }
        // Default region — always present.
        indent(os, indentLevel + 1);
        os << "else => {\n";
        if (failed(emitFuncBody(sw.getDefaultRegion(), os, indentLevel + 2,
                                outerMap, nameMap, tempCount)))
          return failure();
        indent(os, indentLevel + 1);
        os << "}\n";
        indent(os, indentLevel);
        os << "}\n";
        continue;
      }

      // Unary float negation
      if (auto negOp = dyn_cast<arith::NegFOp>(&op)) {
        std::string a = resolve(nameMap, negOp.getOperand());
        std::string tname = "t" + std::to_string(tempCount++);
        indent(os, indentLevel);
        os << "var " << tname << ": " << cslTypeName(negOp.getResult().getType())
           << " = -" << a << ";\n";
        nameMap[negOp.getResult()] = tname;
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

      // csl.builtin_call [in %mod] "name"(args) → `@name(args);` or `mod.name(args);`.
      // Zero results → bare stmt; one result → `const rN = …;`.
      if (auto bc = dyn_cast<xilinx::csl::BuiltinCallOp>(&op)) {
        if (bc.getNumResults() > 1) {
          op.emitError("csl.builtin_call: multi-result not supported in v5");
          return failure();
        }
        indent(os, indentLevel);
        std::string rname;
        if (bc.getNumResults() == 1) {
          rname = "r" + std::to_string(tempCount++);
          os << "const " << rname << " = ";
          nameMap[bc.getResult(0)] = rname;
        }
        if (Value modVal = bc.getModule()) {
          std::string modName = resolve(outerMap, modVal);
          if (modName == "?")
            modName = resolve(nameMap, modVal);
          os << modName << "." << bc.getCallee();
        } else if (bc.getCallee() == "unblock_cmd_stream") {
          // SDK memcpy builtin lives on the implicitly-imported `sys_mod`
          // (see ProgramEmitter preamble). The lower-dataflow-data pass
          // synthesizes csl.builtin_call "unblock_cmd_stream" with no
          // module operand because sys_mod is invented at emit time.
          os << "sys_mod." << bc.getCallee();
        } else {
          os << "@" << bc.getCallee();
        }
        os << "(";
        bool isModuleCall = (bool)bc.getModule();
        for (auto it : llvm::enumerate(bc.getArgs())) {
          if (it.index()) os << ", ";
          std::string argName = resolve(nameMap, it.value());
          // If the value is not in nameMap (e.g., an arith.constant hoisted
          // out of the csl.func body by the pattern rewriter), fall back to
          // emitting the literal inline from the defining op.
          if (argName == "?") {
            if (auto *defOp = it.value().getDefiningOp()) {
              if (auto constOp = dyn_cast<arith::ConstantOp>(defOp)) {
                if (auto floatAttr =
                        dyn_cast<FloatAttr>(constOp.getValue()))
                  argName = std::to_string(floatAttr.getValueAsDouble());
                else if (auto intAttr =
                             dyn_cast<IntegerAttr>(constOp.getValue()))
                  argName = std::to_string(intAttr.getInt());
              }
            }
          }
          // Module-member calls (e.g. `math.sqrt(x)`) take runtime-typed
          // parameters. Our arith.constant lowering leaves literals unnamed
          // in nameMap, so CSL sees `comptime_float` and refuses. Wrap
          // numeric-literal args in `@as(<T>, ...)` to pin the runtime type.
          if (isModuleCall && isNumericLiteral(argName)) {
            os << "@as(" << cslTypeName(it.value().getType()) << ", "
               << argName << ")";
          } else {
            os << argName;
          }
        }
        // Async / activate fields: emit the trailing CSL struct
        //   .{ .async = true, .activate = <activate>_id }
        // The `_id` suffix matches the local-task-id symbol the csl.task
        // emitter generates alongside the task body.
        bool isAsync = bc.getAsync();
        auto activate = bc.getActivateAttr();
        if (isAsync || activate) {
          if (!bc.getArgs().empty())
            os << ", ";
          os << ".{";
          bool first = true;
          if (isAsync) {
            os << " .async = true";
            first = false;
          }
          if (activate) {
            if (!first) os << ",";
            os << " .activate = " << activate.getValue() << "_id";
            first = false;
          }
          os << " }";
        }
        os << ");\n";
        continue;
      }

      // memref.subview / memref.reinterpret_cast — no direct emission.
      // A downstream csl.get_mem_dsd reads shape + strided layout from the
      // result type; the only piece not on the type is the *original*
      // buffer's CSL name (for `.base_address = &X`), so we follow the
      // chain back to the source via nameMap.
      if (auto sv = dyn_cast<memref::SubViewOp>(&op)) {
        Value src = sv.getSource();
        std::string name = resolve(outerMap, src);
        if (name == "?") name = resolve(nameMap, src);
        nameMap[sv.getResult()] = name;
        continue;
      }
      if (auto rc = dyn_cast<memref::ReinterpretCastOp>(&op)) {
        Value src = rc.getSource();
        std::string name = resolve(outerMap, src);
        if (name == "?") name = resolve(nameMap, src);
        nameMap[rc.getResult()] = name;
        continue;
      }

      // csl.get_mem_dsd — reads shape, stride, offset from the operand
      // memref's type. Rank 1 → mem1d_dsd, rank 2..4 → mem4d_dsd.
      // Default identity layouts fold to `stride = 1`, `offset = 0`, both
      // of which are suppressed in emission. Strided views are supplied by
      // upstream memref.subview / memref.reinterpret_cast (handled above).
      //
      // Dynamic offset / size / stride: resolve via the defining op's
      // mixed-form accessors so we emit the SSA name as a CSL variable ref.
      if (auto dsdOp = dyn_cast<xilinx::csl::GetMemDsdOp>(&op)) {
        Value buffer = dsdOp.getBuffer();
        auto memTy = dyn_cast<MemRefType>(buffer.getType());
        if (!memTy) {
          op.emitError("csl.get_mem_dsd expects a memref operand");
          return failure();
        }

        SmallVector<int64_t> strides;
        int64_t offset = 0;
        if (failed(memTy.getStridesAndOffset(strides, offset))) {
          op.emitError("csl.get_mem_dsd: memref must have a strided layout");
          return failure();
        }
        ArrayRef<int64_t> shape = memTy.getShape();
        unsigned rank = shape.size();
        if (rank == 0 || rank > 4) {
          op.emitError("csl.get_mem_dsd: rank must be 1..4");
          return failure();
        }

        // Pull dynamic offsets/sizes/strides from the defining subview/cast,
        // if present. Index i of each mixed vector matches the memref's dim.
        SmallVector<OpFoldResult> mOffs, mSizes, mStrides;
        if (auto sv = buffer.getDefiningOp<memref::SubViewOp>()) {
          mOffs    = sv.getMixedOffsets();
          mSizes   = sv.getMixedSizes();
          mStrides = sv.getMixedStrides();
        } else if (auto rc = buffer.getDefiningOp<memref::ReinterpretCastOp>()) {
          mOffs    = rc.getMixedOffsets();
          mSizes   = rc.getMixedSizes();
          mStrides = rc.getMixedStrides();
        }

        auto foldToStr = [&](OpFoldResult r) -> std::string {
          if (auto a = dyn_cast<Attribute>(r))
            return std::to_string(cast<IntegerAttr>(a).getInt());
          return resolve(nameMap, cast<Value>(r));
        };
        auto staticToStr = [&](int64_t v, size_t idx,
                               ArrayRef<OpFoldResult> dyn) -> std::string {
          if (v != ShapedType::kDynamic) return std::to_string(v);
          if (idx < dyn.size()) return foldToStr(dyn[idx]);
          return "?";
        };

        // base_address is the ORIGINAL buffer's name (follow subview/cast).
        std::string bufName = resolve(outerMap, buffer);
        if (bufName == "?") bufName = resolve(nameMap, buffer);

        // Offset: static from layout, or mixed[0] from the subview.
        //
        // Emitted as `@increment_dsd_offset(<base>, N, <elem_type>)` rather
        // than the `.offset = N` field of @get_dsd. Why: the underlying DSD
        // offset is stored in **16-bit-word units**, not element units —
        // DSDs.md:711-714 says of @increment_dsd_offset's elem_type "is used
        // to convert offset into number of words." Therefore `.offset = N`
        // sets N words directly, and an f32 element-offset of 5 (`.offset=5`,
        // 10 bytes) is misaligned for an f32 access.
        //   Verified at runtime: emitting `.offset = 5` on an f32 buffer for
        //   pick_col made the SDK simulator raise
        //     "MEM[0711] is not 32-bit aligned"
        //   on the first DSD read. `pick_row` (offset=48 → 96 bytes, aligned)
        //   passed only by accident. The `[10]i16` example at DSDs.md:127
        //   does not disambiguate the unit because i16 elements ARE words.
        // `@increment_dsd_offset` does the elements→words conversion for us.
        // Pointer arithmetic `&buf + N` is also forbidden (Types.md:508).
        std::string offsetStr;
        if (offset != ShapedType::kDynamic && offset != 0) {
          offsetStr = std::to_string(offset);
        } else if (offset == ShapedType::kDynamic && !mOffs.empty()) {
          offsetStr = foldToStr(mOffs[0]);
        }
        bool hasOffset = !offsetStr.empty() && offsetStr != "0";

        std::string dname = "d" + std::to_string(tempCount++);
        std::string baseName = hasOffset ? dname + "_base" : dname;
        const char *kind = (rank == 1) ? "mem1d_dsd" : "mem4d_dsd";

        auto emitShape = [&]() {
          if (rank == 1) {
            os << ", .extent = " << staticToStr(shape[0], 0, mSizes);
            std::string s = (!strides.empty())
                ? staticToStr(strides[0], 0, mStrides) : std::string("1");
            if (s != "1") os << ", .stride = " << s;
          } else {
            os << ", .extent = .{";
            for (unsigned i = 0; i < rank; ++i) {
              if (i) os << ", ";
              os << staticToStr(shape[i], i, mSizes);
            }
            os << "}";
            os << ", .stride = .{";
            for (unsigned i = 0; i < rank; ++i) {
              if (i) os << ", ";
              os << staticToStr(strides[i], i, mStrides);
            }
            os << "}";
          }
        };

        indent(os, indentLevel);
        os << "const " << baseName << " = @get_dsd(" << kind
           << ", .{ .base_address = &" << bufName;
        emitShape();
        os << " });\n";

        if (hasOffset) {
          std::string elemTy = cslTypeName(memTy.getElementType());
          indent(os, indentLevel);
          os << "const " << dname << " = @increment_dsd_offset("
             << baseName << ", " << offsetStr << ", " << elemTy << ");\n";
        }

        nameMap[dsdOp.getResult()] = dname;
        continue;
      }

      // csl.get_fab_dsd <dir> @color extent(%n) → fabin/fabout DSD on a color.
      // Emits:
      //   const dN = @get_dsd(<dir>_dsd,
      //                       .{ .extent = N, .fabric_color = <color>,
      //                          .input_queue|.output_queue = <color>_<dir>_q });
      // Extent resolves to either the SSA name in `nameMap` or, if unresolved,
      // the numeric literal from the defining arith.constant. Synthesized by
      // --csl-lower-dataflow-data from csl.dataflow.put/get.
      //
      // The queue field is required on WSE-3; the queue itself is declared
      // at PE file scope (see CSLProgramEmitter), keyed off the same color
      // symbol + direction. WSE-2 ignores the queue field cleanly so we
      // emit it unconditionally for both archs (forward-compat). If it ever
      // becomes a WSE-2 issue, gate the field on a wafer arch check.
      if (auto fab = dyn_cast<xilinx::csl::GetFabDsdOp>(&op)) {
        Value extentVal = fab.getExtent();
        std::string extentStr = resolve(nameMap, extentVal);
        if (extentStr == "?") {
          if (auto *defOp = extentVal.getDefiningOp()) {
            if (auto cst = dyn_cast<arith::ConstantOp>(defOp)) {
              if (auto ia = dyn_cast<IntegerAttr>(cst.getValue()))
                extentStr = std::to_string(ia.getInt());
            }
          }
        }
        StringRef dirStr =
            ::xilinx::csl::stringifyFabDsdDirection(fab.getDirection());
        std::string dname = "d" + std::to_string(tempCount++);
        indent(os, indentLevel);
        std::string colorName = fab.getColor().str();
        std::string qField =
            (dirStr == "fabout") ? ".output_queue = " + colorName + "_out_q"
                                 : ".input_queue = " + colorName + "_in_q";
        os << "const " << dname << " = @get_dsd(" << dirStr
           << "_dsd, .{ .extent = " << extentStr
           << ", .fabric_color = " << colorName
           << ", " << qField
           << " });\n";
        nameMap[fab.getResult()] = dname;
        continue;
      }

      // func.call → var tN: T = callee(args);  (or plain callee(args); if void)
      if (auto callOp = dyn_cast<func::CallOp>(&op)) {
        indent(os, indentLevel);
        std::string tname;
        if (callOp.getNumResults() > 0) {
          tname = "t" + std::to_string(tempCount++);
          os << "var " << tname << ": "
             << cslTypeName(callOp.getResult(0).getType()) << " = ";
          nameMap[callOp.getResult(0)] = tname;
        }
        os << callOp.getCallee() << "(";
        for (auto it : llvm::enumerate(callOp.getOperands())) {
          if (it.index()) os << ", ";
          os << resolve(nameMap, it.value());
        }
        os << ");\n";
        continue;
      }
      // func.return → return [val];
      if (auto retOp = dyn_cast<func::ReturnOp>(&op)) {
        indent(os, indentLevel);
        if (retOp.getNumOperands() > 0)
          os << "return " << resolve(nameMap, retOp.getOperand(0)) << ";\n";
        else
          os << "return;\n";
        continue;
      }

      // csl.get_x_coord → const tN: u16 = layout_mod.get_x_coord();
      // Note: the SDK's layout_mod.get_x_coord() returns u16, so we declare
      // the CSL local as u16 regardless of the MLIR result type (i16).
      if (auto gx = dyn_cast<xilinx::csl::GetXCoordOp>(&op)) {
        std::string tname = "t" + std::to_string(tempCount++);
        indent(os, indentLevel);
        os << "const " << tname << ": u16 = layout_mod.get_x_coord();\n";
        nameMap[gx.getResult()] = tname;
        continue;
      }

      // csl.get_y_coord → const tN: u16 = layout_mod.get_y_coord();
      if (auto gy = dyn_cast<xilinx::csl::GetYCoordOp>(&op)) {
        std::string tname = "t" + std::to_string(tempCount++);
        indent(os, indentLevel);
        os << "const " << tname << ": u16 = layout_mod.get_y_coord();\n";
        nameMap[gy.getResult()] = tname;
        continue;
      }

      // arith.select (bool ? T : F) → CSL `if (cond) T else F` expression.
      // Emits as `var tN: <ty> = if (cond) true_val else false_val;`
      if (auto sel = dyn_cast<arith::SelectOp>(&op)) {
        std::string c = resolve(nameMap, sel.getCondition());
        std::string t = resolve(nameMap, sel.getTrueValue());
        std::string f = resolve(nameMap, sel.getFalseValue());
        std::string tname = "t" + std::to_string(tempCount++);
        indent(os, indentLevel);
        os << "var " << tname << ": "
           << cslTypeName(sel.getResult().getType()) << " = if (" << c
           << ") " << t << " else " << f << ";\n";
        nameMap[sel.getResult()] = tname;
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
