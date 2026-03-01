//===- CSLToTextTranslation.cpp - CSL dialect to CSL text -*- C++ -*-===//
//
// Part of the air-to-csl project.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
//
// Translates CSL dialect IR to Cerebras CSL source files.
// Registered as --emit-csl in air-translate.
//
// Usage:
//   air-translate --emit-csl --csl-output-dir=./output input.mlir
//
// Output files:
//   - For each csl.module @name:        <output-dir>/name.csl
//   - For each csl.kernel "name.csl":   <output-dir>/name.csl  (PE program)
//   - For csl.spatial_placement:        <output-dir>/layout.py  (Python SdkLayout)
//                                       <output-dir>/run.py     (Python host runtime)
//
//===----------------------------------------------------------------------===//

#include "AIRTargets.h"

#include "air/Dialect/CSL/CSLDialect.h"
#include "air/Dialect/CSL/CSLOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Tools/mlir-translate/Translation.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

#include <string>

using namespace mlir;

namespace xilinx {
namespace csl {

//===----------------------------------------------------------------------===//
// CSLWriter — reusable CSL code generation utility
//===----------------------------------------------------------------------===//

class CSLWriter {
public:
  explicit CSLWriter(raw_ostream &os) : os(os) {}

  // -----------------------------------------------------------------------
  // Indentation
  // -----------------------------------------------------------------------

  void indent() { indentLevel++; }
  void dedent() {
    if (indentLevel > 0)
      indentLevel--;
  }

  raw_ostream &line() {
    os << std::string(indentLevel * 2, ' ');
    return os;
  }

  void blankLine() { os << "\n"; }
  void comment(StringRef text) { line() << "// " << text << "\n"; }
  raw_ostream &raw() { return os; }

  // -----------------------------------------------------------------------
  // Type mapping: MLIR types -> CSL type names
  // -----------------------------------------------------------------------

  static std::string mapType(Type ty) {
    if (ty.isF32())
      return "f32";
    if (ty.isF16())
      return "f16";
    if (ty.isBF16())
      return "bf16";
    if (auto intTy = dyn_cast<IntegerType>(ty)) {
      unsigned w = intTy.getWidth();
      bool isSigned = !intTy.isUnsignedInteger();
      if (w == 1)
        return "bool";
      if (w == 64)
        return isSigned ? "i64" : "u64";
      if (w == 32)
        return isSigned ? "i32" : "u32";
      if (w == 16)
        return isSigned ? "i16" : "u16";
      if (w == 8)
        return isSigned ? "i8" : "u8";
    }
    if (ty.isIndex())
      return "i32";
    return "";
  }

  static bool isSupportedType(Type ty) { return !mapType(ty).empty(); }

  // Map a MemRefType to a CSL array type string: e.g. "[1024]f32", "[4, 4]f32"
  static std::string mapMemRefType(MemRefType memTy) {
    std::string elemTy = mapType(memTy.getElementType());
    if (elemTy.empty())
      return "";
    std::string result;
    for (int64_t dim : memTy.getShape()) {
      if (ShapedType::isDynamic(dim))
        result += "[/*dynamic*/]";
      else
        result += "[" + std::to_string(dim) + "]";
    }
    result += elemTy;
    return result;
  }

  static int64_t flatSize(MemRefType memTy) {
    int64_t total = 1;
    for (int64_t dim : memTy.getShape()) {
      if (ShapedType::isDynamic(dim))
        return -1;
      total *= dim;
    }
    return total;
  }

  static std::string pointerType(StringRef elemType) {
    return "[*]" + elemType.str();
  }

  // -----------------------------------------------------------------------
  // Declarations
  // -----------------------------------------------------------------------

  void emitParam(StringRef name, StringRef type) {
    line() << "param " << name << ": " << type << ";\n";
  }

  void emitVar(StringRef name, StringRef type) {
    line() << "var " << name << ": " << type << ";\n";
  }

  void emitVarInit(StringRef name, StringRef type, StringRef init) {
    line() << "var " << name << ": " << type << " = " << init << ";\n";
  }

  void emitConst(StringRef name, StringRef type, StringRef value) {
    line() << "const " << name << ": " << type << " = " << value << ";\n";
  }

  void emitConstInferred(StringRef name, StringRef value) {
    line() << "const " << name << " = " << value << ";\n";
  }

  void emitImportModule(StringRef varName, StringRef modulePath,
                        StringRef paramsExpr = "") {
    line() << "const " << varName << " = @import_module(\"" << modulePath
           << "\"";
    if (!paramsExpr.empty())
      os << ", " << paramsExpr;
    os << ");\n";
  }

  // -----------------------------------------------------------------------
  // Layout constructs
  // -----------------------------------------------------------------------

  void emitLayoutBegin() {
    line() << "layout {\n";
    indent();
  }
  void emitLayoutEnd() {
    dedent();
    line() << "}\n";
  }

  void emitSetRectangle(uint64_t w, uint64_t h) {
    line() << "@set_rectangle(" << w << ", " << h << ");\n";
  }

  void emitSetTileCode(uint64_t col, uint64_t row, StringRef file,
                       StringRef params) {
    line() << "@set_tile_code(" << col << ", " << row << ", \"" << file
           << "\", " << params << ");\n";
  }

  void emitExportName(StringRef name, StringRef type, bool isMutable) {
    line() << "@export_name(\"" << name << "\", " << type << ", "
           << (isMutable ? "true" : "false") << ");\n";
  }

  void emitExportNameFn(StringRef name, StringRef fnSig) {
    line() << "@export_name(\"" << name << "\", " << fnSig << ");\n";
  }

  // -----------------------------------------------------------------------
  // Functions and tasks
  // -----------------------------------------------------------------------

  void emitFuncBegin(StringRef name, StringRef retType = "void") {
    line() << "fn " << name << "() " << retType << " {\n";
    indent();
  }

  void emitFuncEnd() {
    dedent();
    line() << "}\n";
  }

  void emitTaskBegin(StringRef name, int32_t colorId) {
    line() << "task " << name << "() color(" << colorId << ") {\n";
    indent();
  }

  void emitTaskEnd() {
    dedent();
    line() << "}\n";
  }

  // -----------------------------------------------------------------------
  // Comptime block
  // -----------------------------------------------------------------------

  void emitComptimeBegin() {
    line() << "comptime {\n";
    indent();
  }

  void emitComptimeEnd() {
    dedent();
    line() << "}\n";
  }

  void emitExportSymbol(StringRef sym, StringRef alias = "") {
    if (alias.empty())
      line() << "@export_symbol(" << sym << ");\n";
    else
      line() << "@export_symbol(" << sym << ", \"" << alias << "\");\n";
  }

  // -----------------------------------------------------------------------
  // Data movement (DSD / bulk move)
  // -----------------------------------------------------------------------

  void emitMem1dDSD(StringRef name, StringRef baseAddr, StringRef extent) {
    line() << "const " << name
           << " = @get_dsd(mem1d_dsd, .{ .base_address = " << baseAddr
           << ", .extent = " << extent << " });\n";
  }

  void emitFabinDSD(StringRef name, StringRef extent, StringRef color,
                    StringRef queue = "0") {
    line() << "const " << name
           << " = @get_dsd(fabin_dsd, .{ .extent = " << extent
           << ", .fabric_color = " << color << ", .input_queue = " << queue
           << " });\n";
  }

  void emitFaboutDSD(StringRef name, StringRef extent, StringRef color,
                     StringRef queue = "0") {
    line() << "const " << name
           << " = @get_dsd(fabout_dsd, .{ .extent = " << extent
           << ", .fabric_color = " << color << ", .output_queue = " << queue
           << " });\n";
  }

  void emitMov32(StringRef dst, StringRef src) {
    line() << "@mov32(" << dst << ", " << src << ");\n";
  }

  void emitFmovs(StringRef dst, StringRef src) {
    line() << "@fmovs(" << dst << ", " << src << ");\n";
  }

  // -----------------------------------------------------------------------
  // Control flow
  // -----------------------------------------------------------------------

  void emitForRange(StringRef iv, StringRef elemType, StringRef start,
                    StringRef stop, StringRef step) {
    line() << "for (@range(" << elemType << ", " << start << ", " << stop
           << ", " << step << ")) |" << iv << "| {\n";
    indent();
  }

  void emitWhileBegin(StringRef cond, StringRef update) {
    line() << "while (" << cond << ") : (" << update << ") {\n";
    indent();
  }

  void emitLoopEnd() {
    dedent();
    line() << "}\n";
  }

  void emitIfBegin(StringRef cond) {
    line() << "if (" << cond << ") {\n";
    indent();
  }

  void emitElse() {
    dedent();
    line() << "} else {\n";
    indent();
  }

  void emitIfEnd() {
    dedent();
    line() << "}\n";
  }

  // -----------------------------------------------------------------------
  // Expressions and statements
  // -----------------------------------------------------------------------

  void emitBinaryOp(StringRef name, StringRef type, StringRef lhs,
                    StringRef op_str, StringRef rhs) {
    line() << "var " << name << ": " << type << " = " << lhs << " " << op_str
           << " " << rhs << ";\n";
  }

  void emitConstBinaryOp(StringRef name, StringRef type, StringRef lhs,
                         StringRef op_str, StringRef rhs) {
    line() << "const " << name << ": " << type << " = " << lhs << " " << op_str
           << " " << rhs << ";\n";
  }

  void emitArrayAccess(StringRef name, StringRef type, StringRef array,
                       StringRef index) {
    line() << "const " << name << ": " << type << " = " << array << "["
           << index << "];\n";
  }

  void emitArrayStore(StringRef array, StringRef index, StringRef value) {
    line() << array << "[" << index << "] = " << value << ";\n";
  }

  void emitAssign(StringRef target, StringRef value) {
    line() << target << " = " << value << ";\n";
  }

  void emitCall(StringRef fn, StringRef args = "") {
    line() << fn << "(" << args << ");\n";
  }

  void emitMethodCall(StringRef obj, StringRef method, StringRef args = "") {
    line() << obj << "." << method << "(" << args << ");\n";
  }

  void emitAs(StringRef name, StringRef type, StringRef value) {
    line() << "const " << name << " = @as(" << type << ", " << value << ");\n";
  }

  void emitSelect(StringRef name, StringRef type, StringRef cond,
                  StringRef trueVal, StringRef falseVal) {
    line() << "const " << name << ": " << type << " = if (" << cond << ") "
           << trueVal << " else " << falseVal << ";\n";
  }

  void emitNeg(StringRef name, StringRef type, StringRef operand) {
    line() << "var " << name << ": " << type << " = -" << operand << ";\n";
  }

private:
  raw_ostream &os;
  unsigned indentLevel = 0;
};

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

namespace {

// Linearize a multi-dimensional index for CSL array access.
static std::string linearizeIndex(ArrayRef<std::string> idxStrs,
                                  ArrayRef<int64_t> shape) {
  if (idxStrs.empty())
    return "0";
  if (idxStrs.size() == 1)
    return idxStrs[0];
  std::string result;
  llvm::raw_string_ostream ss(result);
  bool first = true;
  for (unsigned d = 0; d < idxStrs.size(); ++d) {
    if (!first)
      ss << " + ";
    ss << idxStrs[d];
    for (unsigned k = d + 1; k < shape.size(); ++k)
      ss << " * " << shape[k];
    first = false;
  }
  return result;
}

// Convert a Direction enum value to its Route.X name for Python output.
static StringRef directionToRouteName(Direction dir) {
  switch (dir) {
  case Direction::NORTH: return "Route.NORTH";
  case Direction::SOUTH: return "Route.SOUTH";
  case Direction::EAST:  return "Route.EAST";
  case Direction::WEST:  return "Route.WEST";
  case Direction::RAMP:  return "Route.RAMP";
  }
  return "Route.RAMP";
}

// Convert an Edge enum value to its Edge.X name for Python output.
static StringRef edgeToName(Edge edge) {
  switch (edge) {
  case Edge::LEFT:   return "Edge.LEFT";
  case Edge::RIGHT:  return "Edge.RIGHT";
  case Edge::TOP:    return "Edge.TOP";
  case Edge::BOTTOM: return "Edge.BOTTOM";
  }
  return "Edge.LEFT";
}

//===----------------------------------------------------------------------===//
// CSLTextEmitter — main translation class
//===----------------------------------------------------------------------===//

class CSLTextEmitter {
public:
  explicit CSLTextEmitter(StringRef outputDir) : outputDir(outputDir) {}

  LogicalResult translate(mlir::ModuleOp module, raw_ostream &diagOS);

private:
  std::string outputDir;
  unsigned nameCounter = 0;

  std::string freshName(StringRef prefix = "v") {
    return (prefix + llvm::Twine(nameCounter++)).str();
  }

  // Write a CSL file to the output directory.
  LogicalResult writeToFile(StringRef filename,
                            llvm::function_ref<LogicalResult(CSLWriter &)> fn,
                            raw_ostream &diagOS);

  // Write a Python file to the output directory.
  LogicalResult writePyToFile(StringRef filename,
                              llvm::function_ref<LogicalResult(raw_ostream &)> fn,
                              raw_ostream &diagOS);

  // Emit the body of a csl.module or csl.kernel (top-level declarations).
  LogicalResult emitModuleBody(CSLWriter &w, Region &body,
                               DenseMap<Value, std::string> &valueNames);

  // Emit the body of a csl.func or csl.task (statements).
  LogicalResult emitFuncBody(CSLWriter &w, Region &body,
                             DenseMap<Value, std::string> &valueNames);

  // Dispatch a single op inside a function body.
  void emitOp(CSLWriter &w, Operation *op,
               DenseMap<Value, std::string> &valueNames);

  // Emit a csl.module to its own .csl file.
  LogicalResult emitModule(mlir::ModuleOp mlirModule, csl::ModuleOp modOp,
                           raw_ostream &diagOS);

  // Emit a csl.spatial_placement: kernel .csl files + layout.py + run.py
  LogicalResult emitSpatialPlacement(mlir::ModuleOp mlirModule,
                                     csl::SpatialPlacementOp placement,
                                     raw_ostream &diagOS);

  // Emit one .csl file for a csl.kernel op.
  LogicalResult emitKernelFile(csl::KernelOp kernel, raw_ostream &diagOS);

  // Shared layout body emitter: called by both emitLayoutPy and emitRunPy.
  // Populates pyNames and collects streams for runtime section.
  LogicalResult emitLayoutBody(
      raw_ostream &py,
      Block &body,
      DenseMap<Value, csl::KernelOp> &kernelForRegion,
      DenseMap<Value, std::string> &pyNames,
      unsigned &counter,
      SmallVectorImpl<std::string> &inputStreamNames,
      SmallVectorImpl<std::string> &outputStreamNames);

  // Emit layout.py (Python SdkLayout API, no runtime) for backward compat.
  LogicalResult emitLayoutPy(csl::SpatialPlacementOp placement,
                             DenseMap<Value, csl::KernelOp> &kernelForRegion,
                             raw_ostream &diagOS);

  // Emit run.py (combined layout + compile + SdkRuntime).
  LogicalResult emitRunPy(csl::SpatialPlacementOp placement,
                          DenseMap<Value, csl::KernelOp> &kernelForRegion,
                          raw_ostream &diagOS);
};

//===----------------------------------------------------------------------===//
// File writing helpers
//===----------------------------------------------------------------------===//

LogicalResult
CSLTextEmitter::writeToFile(StringRef filename,
                             llvm::function_ref<LogicalResult(CSLWriter &)> fn,
                             raw_ostream &diagOS) {
  SmallString<256> path(outputDir);
  llvm::sys::path::append(path, filename);
  std::error_code ec;
  llvm::raw_fd_ostream file(path, ec);
  if (ec) {
    llvm::errs() << "Failed to open '" << path << "': " << ec.message()
                 << "\n";
    return failure();
  }
  CSLWriter w(file);
  if (failed(fn(w)))
    return failure();
  diagOS << "Wrote " << path << "\n";
  return success();
}

LogicalResult CSLTextEmitter::writePyToFile(
    StringRef filename,
    llvm::function_ref<LogicalResult(raw_ostream &)> fn,
    raw_ostream &diagOS) {
  SmallString<256> path(outputDir);
  llvm::sys::path::append(path, filename);
  std::error_code ec;
  llvm::raw_fd_ostream file(path, ec);
  if (ec) {
    llvm::errs() << "Failed to open '" << path << "': " << ec.message()
                 << "\n";
    return failure();
  }
  if (failed(fn(file)))
    return failure();
  diagOS << "Wrote " << path << "\n";
  return success();
}

//===----------------------------------------------------------------------===//
// emitModuleBody — emit top-level ops (param, var, import, func, task, comptime)
//===----------------------------------------------------------------------===//

LogicalResult
CSLTextEmitter::emitModuleBody(CSLWriter &w, Region &body,
                               DenseMap<Value, std::string> &valueNames) {
  for (auto &op : body.front()) {
    // csl.param @name : type
    if (auto param = dyn_cast<csl::ParamOp>(op)) {
      Type ty = param.getType();
      std::string typeName;
      if (auto memTy = dyn_cast<MemRefType>(ty)) {
        typeName = CSLWriter::mapMemRefType(memTy);
        if (typeName.empty())
          typeName = "/* unsupported memref type */";
      } else if (isa<csl::ImportedModuleType>(ty)) {
        typeName = "comptime_struct";
      } else if (isa<csl::ColorType>(ty)) {
        typeName = "color";
      } else {
        typeName = CSLWriter::mapType(ty);
        if (typeName.empty())
          typeName = "/* unsupported type */";
      }
      w.emitParam(param.getSymName().str(), typeName);
      continue;
    }

    // csl.var @name : type
    if (auto var = dyn_cast<csl::VarOp>(op)) {
      Type ty = var.getType();
      std::string typeName;
      if (auto memTy = dyn_cast<MemRefType>(ty)) {
        typeName = CSLWriter::mapMemRefType(memTy);
        if (typeName.empty()) {
          w.comment("TODO: unsupported memref type for var '" +
                    var.getSymName().str() + "'");
          continue;
        }
      } else {
        typeName = CSLWriter::mapType(ty);
        if (typeName.empty()) {
          w.comment("TODO: unsupported type for var '" +
                    var.getSymName().str() + "'");
          continue;
        }
      }
      w.emitVar(var.getSymName().str(), typeName);
      continue;
    }

    // csl.import_module "<path>" params(...)
    if (auto imp = dyn_cast<csl::ImportModuleOp>(op)) {
      // Generate a name from the module path (last path component, no angle
      // brackets)
      StringRef path = imp.getModuleName();
      // Strip angle brackets: "<memcpy/memcpy>" -> "memcpy/memcpy"
      if (path.starts_with("<") && path.ends_with(">"))
        path = path.slice(1, path.size() - 1);
      // Use last component as variable name: "memcpy/memcpy" -> "memcpy"
      StringRef varName = path;
      if (auto slash = path.rfind('/'); slash != StringRef::npos)
        varName = path.substr(slash + 1);
      // Avoid name collision by using a fresh name if needed
      std::string name = varName.str();
      if (name.empty())
        name = freshName("mod_");

      std::string paramsExpr;
      if (auto params = imp.getParams()) {
        llvm::raw_string_ostream ss(paramsExpr);
        ss << ".{";
        bool first = true;
        for (auto &attr : params->getValue()) {
          if (!first)
            ss << ", ";
          ss << " ." << attr.getName().getValue() << " = ";
          if (auto intAttr = dyn_cast<IntegerAttr>(attr.getValue()))
            ss << intAttr.getInt();
          else if (auto strAttr = dyn_cast<StringAttr>(attr.getValue()))
            ss << "\"" << strAttr.getValue() << "\"";
          else
            ss << "/* ... */";
          first = false;
        }
        ss << " }";
      }
      w.emitImportModule(name, imp.getModuleName().str(), paramsExpr);
      valueNames[imp.getResult()] = name;
      continue;
    }

    // csl.func @name() { ... }
    if (auto func = dyn_cast<csl::FuncOp>(op)) {
      w.emitFuncBegin(func.getSymName().str());
      DenseMap<Value, std::string> localNames;
      if (failed(emitFuncBody(w, func.getBody(), localNames)))
        return failure();
      w.emitFuncEnd();
      w.blankLine();
      continue;
    }

    // csl.task @name() color(N) { ... }
    if (auto task = dyn_cast<csl::TaskOp>(op)) {
      w.emitTaskBegin(task.getSymName().str(), task.getColorId());
      DenseMap<Value, std::string> localNames;
      if (failed(emitFuncBody(w, task.getBody(), localNames)))
        return failure();
      w.emitTaskEnd();
      w.blankLine();
      continue;
    }

    // csl.comptime { ... }
    if (auto comptime = dyn_cast<csl::ComptimeOp>(op)) {
      w.emitComptimeBegin();
      for (auto &comptimeOp : comptime.getBody().front()) {
        if (auto exportSym = dyn_cast<csl::ExportSymbolOp>(comptimeOp)) {
          std::string alias;
          if (auto aliasAttr = exportSym.getAlias())
            alias = aliasAttr->str();
          w.emitExportSymbol(exportSym.getSym().str(), alias);
        } else if (isa<csl::ExportNameOp>(comptimeOp)) {
          // ExportNameOp more commonly lives at module level (for layout);
          // if inside comptime, emit a comment.
          w.comment(
              "export_name (see layout file for @export_name directives)");
        } else if (!comptimeOp.hasTrait<OpTrait::IsTerminator>()) {
          w.comment("TODO: comptime op: " +
                    comptimeOp.getName().getStringRef().str());
        }
      }
      w.emitComptimeEnd();
      continue;
    }

    // Skip terminators
    if (op.hasTrait<OpTrait::IsTerminator>())
      continue;

    // Fallback
    w.comment("TODO: unsupported top-level op: " +
              op.getName().getStringRef().str());
  }
  return success();
}

//===----------------------------------------------------------------------===//
// emitFuncBody — emit statements inside csl.func or csl.task
//===----------------------------------------------------------------------===//

LogicalResult
CSLTextEmitter::emitFuncBody(CSLWriter &w, Region &body,
                              DenseMap<Value, std::string> &valueNames) {
  if (body.empty())
    return success();
  for (auto &op : body.front())
    emitOp(w, &op, valueNames);
  return success();
}

//===----------------------------------------------------------------------===//
// emitOp — dispatch individual ops in function bodies
//===----------------------------------------------------------------------===//

void CSLTextEmitter::emitOp(CSLWriter &w, Operation *op,
                              DenseMap<Value, std::string> &valueNames) {
  auto getVal = [&](Value v) -> std::string {
    auto it = valueNames.find(v);
    if (it != valueNames.end())
      return it->second;
    return "/* unknown */";
  };

  // ---- CSL dialect ops inside function bodies ----

  // csl.return — implicit, no-op
  if (isa<csl::ReturnOp>(op))
    return;

  // csl.get_mem_dsd %buf, %len
  if (auto memDsd = dyn_cast<csl::GetMemDsdOp>(op)) {
    std::string name = freshName("dsd_");
    w.emitMem1dDSD(name, getVal(memDsd.getBuffer()),
                   getVal(memDsd.getLength()));
    valueNames[memDsd.getResult()] = name;
    return;
  }

  // csl.get_fab_dsd fabin/fabout %c, %len
  if (auto fabDsd = dyn_cast<csl::GetFabDsdOp>(op)) {
    std::string name = freshName("dsd_");
    std::string color = getVal(fabDsd.getColor());
    std::string len = getVal(fabDsd.getLength());
    if (fabDsd.getKind() == DsdKind::fabin)
      w.emitFabinDSD(name, len, color);
    else
      w.emitFaboutDSD(name, len, color);
    valueNames[fabDsd.getResult()] = name;
    return;
  }

  // csl.mov %dst, %src
  if (auto mov = dyn_cast<csl::MovOp>(op)) {
    w.emitMov32(getVal(mov.getDst()), getVal(mov.getSrc()));
    return;
  }

  // csl.color (in function scope — e.g. @get_color(N))
  if (auto colorOp = dyn_cast<csl::ColorOp>(op)) {
    std::string name = freshName("color_");
    if (auto id = colorOp.getId())
      w.emitConst(name, "color",
                  "@get_color(" + std::to_string(*id) + ")");
    else
      w.emitConst(name, "color", "@get_color(/* TODO */)");
    valueNames[colorOp.getResult()] = name;
    return;
  }

  // ---- Standard MLIR ops (arith / memref / scf / func) ----

  // arith.constant
  if (auto cst = dyn_cast<arith::ConstantOp>(op)) {
    std::string cslTy = CSLWriter::mapType(cst.getType());
    if (cslTy.empty()) {
      std::string typeStr;
      llvm::raw_string_ostream(typeStr) << cst.getType();
      w.comment("unsupported constant type: " + typeStr);
      valueNames[cst.getResult()] = "/* unsupported */";
      return;
    }
    std::string name = freshName("c_");
    auto attr = cst.getValue();
    if (auto intAttr = dyn_cast<IntegerAttr>(attr)) {
      w.emitConst(name, cslTy, std::to_string(intAttr.getInt()));
    } else if (auto fpAttr = dyn_cast<FloatAttr>(attr)) {
      SmallString<32> fpStr;
      fpAttr.getValue().toString(fpStr, 6);
      std::string val(fpStr);
      // Ensure floating-point literal has a decimal point
      if (val.find('.') == std::string::npos &&
          val.find('e') == std::string::npos)
        val += ".0";
      w.emitConst(name, cslTy, val);
    } else {
      w.comment("unsupported constant attribute");
      w.emitConst(name, cslTy, "undefined");
    }
    valueNames[cst.getResult()] = name;
    return;
  }

  // arith binary ops (float and integer, all use emitBinaryOp)
  auto emitBinOp = [&](Value result, Value lhs, Value rhs, StringRef opStr) {
    std::string name = freshName("v_");
    w.emitBinaryOp(name, CSLWriter::mapType(result.getType()), getVal(lhs),
                   opStr, getVal(rhs));
    valueNames[result] = name;
  };

  if (auto x = dyn_cast<arith::AddFOp>(op))  { emitBinOp(x.getResult(), x.getLhs(), x.getRhs(), "+");  return; }
  if (auto x = dyn_cast<arith::SubFOp>(op))  { emitBinOp(x.getResult(), x.getLhs(), x.getRhs(), "-");  return; }
  if (auto x = dyn_cast<arith::MulFOp>(op))  { emitBinOp(x.getResult(), x.getLhs(), x.getRhs(), "*");  return; }
  if (auto x = dyn_cast<arith::DivFOp>(op))  { emitBinOp(x.getResult(), x.getLhs(), x.getRhs(), "/");  return; }
  if (auto x = dyn_cast<arith::AddIOp>(op))  { emitBinOp(x.getResult(), x.getLhs(), x.getRhs(), "+");  return; }
  if (auto x = dyn_cast<arith::SubIOp>(op))  { emitBinOp(x.getResult(), x.getLhs(), x.getRhs(), "-");  return; }
  if (auto x = dyn_cast<arith::MulIOp>(op))  { emitBinOp(x.getResult(), x.getLhs(), x.getRhs(), "*");  return; }
  if (auto x = dyn_cast<arith::DivSIOp>(op)) { emitBinOp(x.getResult(), x.getLhs(), x.getRhs(), "/");  return; }
  if (auto x = dyn_cast<arith::RemSIOp>(op)) { emitBinOp(x.getResult(), x.getLhs(), x.getRhs(), "%");  return; }
  if (auto x = dyn_cast<arith::DivUIOp>(op)) { emitBinOp(x.getResult(), x.getLhs(), x.getRhs(), "/");  return; }
  if (auto x = dyn_cast<arith::RemUIOp>(op)) { emitBinOp(x.getResult(), x.getLhs(), x.getRhs(), "%");  return; }
  if (auto x = dyn_cast<arith::AndIOp>(op))  { emitBinOp(x.getResult(), x.getLhs(), x.getRhs(), "&");  return; }
  if (auto x = dyn_cast<arith::OrIOp>(op))   { emitBinOp(x.getResult(), x.getLhs(), x.getRhs(), "|");  return; }
  if (auto x = dyn_cast<arith::XOrIOp>(op))  { emitBinOp(x.getResult(), x.getLhs(), x.getRhs(), "^");  return; }
  if (auto x = dyn_cast<arith::ShLIOp>(op))  { emitBinOp(x.getResult(), x.getLhs(), x.getRhs(), "<<"); return; }
  if (auto x = dyn_cast<arith::ShRSIOp>(op)) { emitBinOp(x.getResult(), x.getLhs(), x.getRhs(), ">>"); return; }
  if (auto x = dyn_cast<arith::ShRUIOp>(op)) { emitBinOp(x.getResult(), x.getLhs(), x.getRhs(), ">>"); return; }

  // arith.negf
  if (auto negf = dyn_cast<arith::NegFOp>(op)) {
    std::string name = freshName("v_");
    w.emitNeg(name, CSLWriter::mapType(negf.getType()),
              getVal(negf.getOperand()));
    valueNames[negf.getResult()] = name;
    return;
  }

  // arith cast ops
  auto emitCast = [&](Value result, Value in) {
    std::string name = freshName("v_");
    w.emitAs(name, CSLWriter::mapType(result.getType()), getVal(in));
    valueNames[result] = name;
  };

  if (auto x = dyn_cast<arith::IndexCastOp>(op)) { emitCast(x.getResult(), x.getIn()); return; }
  if (auto x = dyn_cast<arith::SIToFPOp>(op))    { emitCast(x.getResult(), x.getIn()); return; }
  if (auto x = dyn_cast<arith::FPToSIOp>(op))    { emitCast(x.getResult(), x.getIn()); return; }
  if (auto x = dyn_cast<arith::ExtFOp>(op))      { emitCast(x.getResult(), x.getIn()); return; }
  if (auto x = dyn_cast<arith::TruncFOp>(op))    { emitCast(x.getResult(), x.getIn()); return; }
  if (auto x = dyn_cast<arith::ExtSIOp>(op))     { emitCast(x.getResult(), x.getIn()); return; }
  if (auto x = dyn_cast<arith::TruncIOp>(op))    { emitCast(x.getResult(), x.getIn()); return; }
  if (auto x = dyn_cast<arith::ExtUIOp>(op))     { emitCast(x.getResult(), x.getIn()); return; }

  // arith.select
  if (auto sel = dyn_cast<arith::SelectOp>(op)) {
    std::string name = freshName("v_");
    w.emitSelect(name, CSLWriter::mapType(sel.getType()),
                 getVal(sel.getCondition()), getVal(sel.getTrueValue()),
                 getVal(sel.getFalseValue()));
    valueNames[sel.getResult()] = name;
    return;
  }

  // arith.cmpi
  if (auto cmpi = dyn_cast<arith::CmpIOp>(op)) {
    std::string name = freshName("cmp_");
    StringRef opStr;
    switch (cmpi.getPredicate()) {
    case arith::CmpIPredicate::eq:  opStr = "=="; break;
    case arith::CmpIPredicate::ne:  opStr = "!="; break;
    case arith::CmpIPredicate::slt:
    case arith::CmpIPredicate::ult: opStr = "<";  break;
    case arith::CmpIPredicate::sle:
    case arith::CmpIPredicate::ule: opStr = "<="; break;
    case arith::CmpIPredicate::sgt:
    case arith::CmpIPredicate::ugt: opStr = ">";  break;
    case arith::CmpIPredicate::sge:
    case arith::CmpIPredicate::uge: opStr = ">="; break;
    }
    w.emitConstBinaryOp(name, "bool", getVal(cmpi.getLhs()), opStr,
                        getVal(cmpi.getRhs()));
    valueNames[cmpi.getResult()] = name;
    return;
  }

  // arith.cmpf
  if (auto cmpf = dyn_cast<arith::CmpFOp>(op)) {
    std::string name = freshName("cmp_");
    StringRef opStr;
    switch (cmpf.getPredicate()) {
    case arith::CmpFPredicate::OEQ:
    case arith::CmpFPredicate::UEQ: opStr = "=="; break;
    case arith::CmpFPredicate::ONE:
    case arith::CmpFPredicate::UNE: opStr = "!="; break;
    case arith::CmpFPredicate::OLT:
    case arith::CmpFPredicate::ULT: opStr = "<";  break;
    case arith::CmpFPredicate::OLE:
    case arith::CmpFPredicate::ULE: opStr = "<="; break;
    case arith::CmpFPredicate::OGT:
    case arith::CmpFPredicate::UGT: opStr = ">";  break;
    case arith::CmpFPredicate::OGE:
    case arith::CmpFPredicate::UGE: opStr = ">="; break;
    default: opStr = "=="; break;
    }
    w.emitConstBinaryOp(name, "bool", getVal(cmpf.getLhs()), opStr,
                        getVal(cmpf.getRhs()));
    valueNames[cmpf.getResult()] = name;
    return;
  }

  // memref.load
  if (auto load = dyn_cast<memref::LoadOp>(op)) {
    std::string name = freshName("ld_");
    SmallVector<std::string> idxStrs;
    for (auto idx : load.getIndices())
      idxStrs.push_back(getVal(idx));
    std::string indexExpr =
        linearizeIndex(idxStrs, load.getMemRefType().getShape());
    w.emitArrayAccess(name, CSLWriter::mapType(load.getType()),
                      getVal(load.getMemRef()), indexExpr);
    valueNames[load.getResult()] = name;
    return;
  }

  // memref.store
  if (auto store = dyn_cast<memref::StoreOp>(op)) {
    SmallVector<std::string> idxStrs;
    for (auto idx : store.getIndices())
      idxStrs.push_back(getVal(idx));
    std::string indexExpr =
        linearizeIndex(idxStrs, store.getMemRefType().getShape());
    w.emitArrayStore(getVal(store.getMemRef()), indexExpr,
                     getVal(store.getValueToStore()));
    return;
  }

  // memref.alloc (local buffer in function body)
  if (auto alloc = dyn_cast<memref::AllocOp>(op)) {
    std::string name = freshName("buf_");
    std::string typeName = CSLWriter::mapMemRefType(alloc.getType());
    if (typeName.empty())
      w.comment("TODO: unsupported alloc type");
    else
      w.emitVar(name, typeName);
    valueNames[alloc.getResult()] = name;
    return;
  }

  // memref.dealloc — no-op in CSL
  if (isa<memref::DeallocOp>(op))
    return;

  // scf.for
  if (auto forOp = dyn_cast<scf::ForOp>(op)) {
    std::string iv = freshName("i_");
    valueNames[forOp.getInductionVar()] = iv;

    std::string lb = getVal(forOp.getLowerBound());
    std::string ub = getVal(forOp.getUpperBound());
    std::string step = getVal(forOp.getStep());
    std::string ivType = CSLWriter::mapType(forOp.getInductionVar().getType());

    // Check if step is constant 1 (use CSL @range syntax)
    bool stepIsOne = false;
    if (auto cst = forOp.getStep().getDefiningOp<arith::ConstantOp>())
      if (auto intAttr = dyn_cast<IntegerAttr>(cst.getValue()))
        stepIsOne = (intAttr.getInt() == 1);

    if (stepIsOne) {
      w.emitForRange(iv, ivType, lb, ub, "1");
    } else {
      w.emitVarInit(iv, ivType, lb);
      w.emitWhileBegin(iv + " < " + ub, iv + " += " + step);
    }

    // Emit iter-arg initializers as mutable vars
    for (auto [iterArg, initVal] :
         llvm::zip(forOp.getRegionIterArgs(), forOp.getInitArgs())) {
      std::string accName = freshName("acc_");
      w.emitVarInit(accName, CSLWriter::mapType(iterArg.getType()),
                    getVal(initVal));
      valueNames[iterArg] = accName;
    }

    for (auto &bodyOp : forOp.getBody()->getOperations()) {
      if (auto yieldOp = dyn_cast<scf::YieldOp>(bodyOp)) {
        // Update iter-args at the end of the loop body
        for (auto [iterArg, yieldedVal] :
             llvm::zip(forOp.getRegionIterArgs(), yieldOp.getResults()))
          w.emitAssign(valueNames[iterArg], getVal(yieldedVal));
        continue;
      }
      emitOp(w, &bodyOp, valueNames);
    }

    w.emitLoopEnd();

    // Map for-op results to final iter-arg names
    for (auto [res, iterArg] :
         llvm::zip(forOp.getResults(), forOp.getRegionIterArgs()))
      valueNames[res] = valueNames[iterArg];
    return;
  }

  // scf.if
  if (auto ifOp = dyn_cast<scf::IfOp>(op)) {
    w.emitIfBegin(getVal(ifOp.getCondition()));
    for (auto &bodyOp : ifOp.getThenRegion().front()) {
      if (isa<scf::YieldOp>(bodyOp))
        continue;
      emitOp(w, &bodyOp, valueNames);
    }
    if (!ifOp.getElseRegion().empty()) {
      w.emitElse();
      for (auto &bodyOp : ifOp.getElseRegion().front()) {
        if (isa<scf::YieldOp>(bodyOp))
          continue;
        emitOp(w, &bodyOp, valueNames);
      }
    }
    w.emitIfEnd();
    return;
  }

  // func.call
  if (auto call = dyn_cast<func::CallOp>(op)) {
    std::string args;
    llvm::raw_string_ostream ss(args);
    bool first = true;
    for (auto arg : call.getOperands()) {
      if (!first)
        ss << ", ";
      ss << getVal(arg);
      first = false;
    }
    w.emitCall(call.getCallee().str(), args);
    return;
  }

  // Skip terminators (yield, herd_terminator, etc.)
  if (op->hasTrait<OpTrait::IsTerminator>())
    return;

  // Fallback
  w.comment("TODO: unsupported op: " + op->getName().getStringRef().str());
}

//===----------------------------------------------------------------------===//
// emitModule — emit a csl.module @name { ... } to name.csl
//===----------------------------------------------------------------------===//

LogicalResult CSLTextEmitter::emitModule(mlir::ModuleOp mlirModule,
                                          csl::ModuleOp modOp,
                                          raw_ostream &diagOS) {
  std::string filename = modOp.getSymName().str() + ".csl";
  return writeToFile(
      filename,
      [&](CSLWriter &w) -> LogicalResult {
        w.comment("Auto-generated by air-translate --emit-csl");
        w.blankLine();
        DenseMap<Value, std::string> valueNames;
        return emitModuleBody(w, modOp.getBody(), valueNames);
      },
      diagOS);
}

//===----------------------------------------------------------------------===//
// emitKernelFile — emit one .csl file for a csl.kernel op
//===----------------------------------------------------------------------===//

LogicalResult CSLTextEmitter::emitKernelFile(csl::KernelOp kernel,
                                              raw_ostream &diagOS) {
  StringRef filename = kernel.getSourceFile();
  return writeToFile(
      filename,
      [&](CSLWriter &w) -> LogicalResult {
        w.comment("Auto-generated by air-translate --emit-csl");
        w.blankLine();

        // Emit param declarations from the kernel's params dict
        if (auto params = kernel.getParams()) {
          for (auto &attr : params->getValue()) {
            std::string paramName = attr.getName().getValue().str();
            if (paramName == "memcpy_params") {
              w.emitParam(paramName, "comptime_struct");
              w.emitImportModule("sys_mod", "<memcpy/memcpy>", paramName);
            } else {
              // Infer type from the attribute value
              std::string typeName = "i32";
              if (auto intAttr = dyn_cast<IntegerAttr>(attr.getValue())) {
                unsigned bits =
                    intAttr.getType().getIntOrFloatBitWidth();
                typeName = (bits == 64) ? "i64" : "i32";
              }
              w.emitParam(paramName, typeName);
            }
          }
          w.blankLine();
        }

        // Emit the kernel body (vars, funcs, tasks, comptime, etc.)
        DenseMap<Value, std::string> valueNames;
        return emitModuleBody(w, kernel.getBody(), valueNames);
      },
      diagOS);
}

//===----------------------------------------------------------------------===//
// emitLayoutBody — shared single-pass layout emitter
//
// Iterates over all ops in the spatial_placement body in source order and
// emits the corresponding Python SdkLayout API calls. Both emitLayoutPy and
// emitRunPy call this helper (with the same pyNames map so that SSA values
// resolve correctly).
//===----------------------------------------------------------------------===//

LogicalResult CSLTextEmitter::emitLayoutBody(
    raw_ostream &py,
    Block &body,
    DenseMap<Value, csl::KernelOp> &kernelForRegion,
    DenseMap<Value, std::string> &pyNames,
    unsigned &counter,
    SmallVectorImpl<std::string> &inputStreamNames,
    SmallVectorImpl<std::string> &outputStreamNames) {

  auto freshPyName = [&](StringRef prefix) -> std::string {
    return (prefix + llvm::Twine(counter++)).str();
  };

  auto getPyName = [&](Value v) -> std::string {
    auto it = pyNames.find(v);
    return it != pyNames.end() ? it->second : "/* unknown */";
  };

  auto emitAttr = [&](Attribute attr) {
    if (auto i = dyn_cast<IntegerAttr>(attr))
      py << i.getValue().getSExtValue();
    else if (auto f = dyn_cast<FloatAttr>(attr))
      py << f.getValueAsDouble();
    else
      py << "# unsupported value";
  };

  for (auto &op : body) {
    // csl.color — layout.alloc_color()
    if (auto colorOp = dyn_cast<csl::ColorOp>(op)) {
      std::string name = freshPyName("color_");
      pyNames[colorOp.getResult()] = name;
      if (auto id = colorOp.getId())
        py << name << " = layout.alloc_color(" << *id << ")\n";
      else
        py << name << " = layout.alloc_color()\n";
      continue;
    }

    // csl.sym_color "name" — Color('name')
    if (auto symColor = dyn_cast<csl::SymColorOp>(op)) {
      std::string name = symColor.getSymName().str();
      pyNames[symColor.getResult()] = name;
      py << name << " = Color('" << name << "')\n";
      continue;
    }

    // csl.scoped_color %region "name" — regionPyName.color('name')
    if (auto scopedColor = dyn_cast<csl::ScopedColorOp>(op)) {
      std::string regionName = getPyName(scopedColor.getRegion());
      std::string symName = scopedColor.getSymName().str();
      std::string colorName = regionName + "_" + symName;
      pyNames[scopedColor.getResult()] = colorName;
      py << colorName << " = " << regionName << ".color('" << symName << "')\n";
      continue;
    }

    // csl.route — RoutingPosition() with optional set_input / set_output
    if (auto routeOp = dyn_cast<csl::RouteOp>(op)) {
      std::string name = freshPyName("rp_");
      pyNames[routeOp.getResult()] = name;
      py << name << " = RoutingPosition()\n";
      if (auto inDir = routeOp.getInputDir())
        py << name << ".set_input([" << directionToRouteName(*inDir) << "])\n";
      if (auto outDir = routeOp.getOutputDir())
        py << name << ".set_output([" << directionToRouteName(*outDir) << "])\n";
      continue;
    }

    // csl.code_region — create_code_region + paint ops from body
    if (auto regionOp = dyn_cast<csl::CodeRegionOp>(op)) {
      // Determine python name: use region_name attr if present
      std::string pyName;
      if (auto nameAttr = regionOp.getRegionName())
        pyName = nameAttr->str();
      else
        pyName = freshPyName("code");
      pyNames[regionOp.getResult()] = pyName;

      // Find kernel for filename
      StringRef filename = "pe_program.csl";
      auto it = kernelForRegion.find(regionOp.getResult());
      if (it != kernelForRegion.end())
        filename = it->second.getSourceFile();

      py << pyName << " = layout.create_code_region('./" << filename
         << "', '" << pyName << "', "
         << regionOp.getWidth() << ", " << regionOp.getHeight() << ")\n";

      // Emit paint ops from the region body
      if (!regionOp.getBody().empty()) {
        for (auto &bodyOp : regionOp.getBody().front()) {
          if (auto paintOp = dyn_cast<csl::PaintOp>(bodyOp)) {
            py << pyName << ".paint(IntVector(" << paintOp.getPeX()
               << ", " << paintOp.getPeY() << "), "
               << getPyName(paintOp.getColor()) << ", ["
               << getPyName(paintOp.getRoute()) << "])\n";
          }
        }
      }
      continue;
    }

    // csl.place — regionPyName.place(x, y)
    if (auto placeOp = dyn_cast<csl::PlaceOp>(op)) {
      py << getPyName(placeOp.getRegion()) << ".place("
         << placeOp.getX() << ", " << placeOp.getY() << ")\n";
      continue;
    }

    // csl.set_param_all — regionPyName.set_param_all('name', value)
    if (auto setParam = dyn_cast<csl::SetParamAllOp>(op)) {
      py << getPyName(setParam.getRegion()) << ".set_param_all('"
         << setParam.getParamName() << "', ";
      emitAttr(setParam.getValue());
      py << ")\n";
      continue;
    }

    // csl.set_param — regionPyName.set_param(IntVector(x, y), 'name', value)
    if (auto setParam = dyn_cast<csl::SetParamOp>(op)) {
      py << getPyName(setParam.getRegion()) << ".set_param(IntVector("
         << setParam.getPeX() << ", " << setParam.getPeY() << "), '"
         << setParam.getParamName() << "', ";
      emitAttr(setParam.getValue());
      py << ")\n";
      continue;
    }

    // csl.set_param_color — regionPyName.set_param_all(colorPyName)
    if (auto setParamColor = dyn_cast<csl::SetParamColorOp>(op)) {
      py << getPyName(setParamColor.getRegion()) << ".set_param_all("
         << getPyName(setParamColor.getColor()) << ")\n";
      continue;
    }

    // csl.routing_position — RoutingPosition().set_input([...]).set_output([...])
    if (auto rpOp = dyn_cast<csl::RoutingPositionOp>(op)) {
      std::string name = freshPyName("rp_");
      pyNames[rpOp.getResult()] = name;
      py << name << " = RoutingPosition()";
      auto emitDirs = [&](StringRef dirsStr, StringRef method) {
        py << "." << method.str() << "([";
        SmallVector<StringRef> parts;
        dirsStr.split(parts, ',');
        bool first = true;
        for (auto &d : parts) {
          if (!first)
            py << ", ";
          py << "Route." << d.trim().str();
          first = false;
        }
        py << "])";
      };
      if (auto inDirs = rpOp.getInDirs())
        emitDirs(*inDirs, "set_input");
      if (auto outDirs = rpOp.getOutDirs())
        emitDirs(*outDirs, "set_output");
      py << "\n";
      continue;
    }

    // csl.get_edge_routing — er_N = get_edge_routing(Edge.X, [rp0, ...])
    if (auto erOp = dyn_cast<csl::GetEdgeRoutingOp>(op)) {
      std::string name = freshPyName("er_");
      pyNames[erOp.getResult()] = name;
      py << name << " = get_edge_routing(" << edgeToName(erOp.getEdge()) << ", [";
      bool first = true;
      for (auto route : erOp.getRoutes()) {
        if (!first)
          py << ", ";
        py << getPyName(route);
        first = false;
      }
      py << "])\n";
      continue;
    }

    // csl.paint_all — regionName.paint_all(color, [core_routes], [edge_routes])
    if (auto paintAll = dyn_cast<csl::PaintAllOp>(op)) {
      std::string regionName = getPyName(paintAll.getRegion());
      py << regionName << ".paint_all(" << getPyName(paintAll.getColor()) << ", [";
      bool first = true;
      for (auto route : paintAll.getCoreRoutes()) {
        if (!first)
          py << ", ";
        py << getPyName(route);
        first = false;
      }
      py << "]";
      if (!paintAll.getEdgeRoutes().empty()) {
        py << ", [";
        first = true;
        for (auto route : paintAll.getEdgeRoutes()) {
          if (!first)
            py << ", ";
          py << getPyName(route);
          first = false;
        }
        py << "]";
      }
      py << ")\n";
      continue;
    }

    // csl.paint_range — regionName.paint_range(IntRectangle(...), color, [routes])
    if (auto pr = dyn_cast<csl::PaintRangeOp>(op)) {
      std::string regionName = getPyName(pr.getRegion());
      py << regionName << ".paint_range(IntRectangle(IntVector("
         << pr.getUlX() << ", " << pr.getUlY() << "), IntVector("
         << pr.getLrX() << ", " << pr.getLrY() << ")), "
         << getPyName(pr.getColor()) << ", [";
      bool first = true;
      for (auto route : pr.getRoutes()) {
        if (!first)
          py << ", ";
        py << getPyName(route);
        first = false;
      }
      py << "])\n";
      continue;
    }

    // csl.set_param_range_color — regionName.set_param_range(IntRectangle(...), color)
    if (auto spr = dyn_cast<csl::SetParamRangeColorOp>(op)) {
      std::string regionName = getPyName(spr.getRegion());
      py << regionName << ".set_param_range(IntRectangle(IntVector("
         << spr.getUlX() << ", " << spr.getUlY() << "), IntVector("
         << spr.getLrX() << ", " << spr.getLrY() << ")), "
         << getPyName(spr.getColor()) << ")\n";
      continue;
    }

    // csl.set_param_range_named — regionName.set_param_range(IntRectangle(...), 'name', color)
    if (auto sprn = dyn_cast<csl::SetParamRangeNamedOp>(op)) {
      std::string regionName = getPyName(sprn.getRegion());
      py << regionName << ".set_param_range(IntRectangle(IntVector("
         << sprn.getUlX() << ", " << sprn.getUlY() << "), IntVector("
         << sprn.getLrX() << ", " << sprn.getLrY() << ")), '"
         << sprn.getParamName().str() << "', "
         << getPyName(sprn.getColor()) << ")\n";
      continue;
    }

    // csl.input_port — portName = regionPyName.create_input_port(...)
    if (auto inPort = dyn_cast<csl::InputPortOp>(op)) {
      std::string name = freshPyName("port");
      pyNames[inPort.getResult()] = name;
      py << name << " = " << getPyName(inPort.getRegion())
         << ".create_input_port(" << getPyName(inPort.getColor())
         << ", " << edgeToName(inPort.getEdge()) << ", [";
      bool first = true;
      for (auto route : inPort.getRoutes()) {
        if (!first)
          py << ", ";
        py << getPyName(route);
        first = false;
      }
      py << "], " << inPort.getSize() << ")\n";
      continue;
    }

    // csl.output_port — portName = regionPyName.create_output_port(...)
    if (auto outPort = dyn_cast<csl::OutputPortOp>(op)) {
      std::string name = freshPyName("port");
      pyNames[outPort.getResult()] = name;
      py << name << " = " << getPyName(outPort.getRegion())
         << ".create_output_port(" << getPyName(outPort.getColor())
         << ", " << edgeToName(outPort.getEdge()) << ", [";
      bool first = true;
      for (auto route : outPort.getRoutes()) {
        if (!first)
          py << ", ";
        py << getPyName(route);
        first = false;
      }
      py << "], " << outPort.getSize() << ")\n";
      continue;
    }

    // csl.dataflow — layout.connect(src, dst)
    if (auto dfOp = dyn_cast<csl::DataflowOp>(op)) {
      py << "layout.connect(" << getPyName(dfOp.getSrc())
         << ", " << getPyName(dfOp.getDst()) << ")\n";
      continue;
    }

    // csl.input_stream — streamName = layout.create_input_stream(port)
    if (auto inStream = dyn_cast<csl::InputStreamOp>(op)) {
      std::string name = freshPyName("in_stream");
      pyNames[inStream.getResult()] = name;
      py << name << " = layout.create_input_stream("
         << getPyName(inStream.getPort()) << ")\n";
      inputStreamNames.push_back(name);
      continue;
    }

    // csl.output_stream — streamName = layout.create_output_stream(port)
    if (auto outStream = dyn_cast<csl::OutputStreamOp>(op)) {
      std::string name = freshPyName("out_stream");
      pyNames[outStream.getResult()] = name;
      py << name << " = layout.create_output_stream("
         << getPyName(outStream.getPort()) << ")\n";
      outputStreamNames.push_back(name);
      continue;
    }

    // csl.export_name — layout.export_name(...)
    if (auto exportName = dyn_cast<csl::ExportNameOp>(op)) {
      Type ty = exportName.getType();
      std::string symName = exportName.getSymName().str();
      if (auto fnTy = dyn_cast<FunctionType>(ty)) {
        (void)fnTy;
        py << "layout.export_name(\"" << symName << "\", \"fn()void\")\n";
      } else if (auto memTy = dyn_cast<MemRefType>(ty)) {
        std::string elemTy = CSLWriter::mapType(memTy.getElementType());
        if (!elemTy.empty()) {
          std::string ptrTy = CSLWriter::pointerType(elemTy);
          py << "layout.export_name(\"" << symName << "\", \""
             << ptrTy << "\", mutable=True)\n";
        } else {
          py << "# layout.export_name(\"" << symName
             << "\", \"/* unsupported type */\")\n";
        }
      } else {
        std::string cslTy = CSLWriter::mapType(ty);
        if (!cslTy.empty())
          py << "layout.export_name(\"" << symName << "\", \"" << cslTy << "\")\n";
        else
          py << "# layout.export_name(\"" << symName
             << "\", \"/* unsupported type */\")\n";
      }
      continue;
    }

    // csl.kernel, csl.port (old), csl.set_param_all already handled above,
    // terminators, and other ops: skip silently.
  }
  return success();
}

//===----------------------------------------------------------------------===//
// emitLayoutPy — emit layout.py (Python SdkLayout API, without runtime)
//===----------------------------------------------------------------------===//

LogicalResult CSLTextEmitter::emitLayoutPy(
    csl::SpatialPlacementOp placement,
    DenseMap<Value, csl::KernelOp> &kernelForRegion,
    raw_ostream &diagOS) {

  return writePyToFile(
      "layout.py",
      [&](raw_ostream &py) -> LogicalResult {
        Block &body = placement.getBody().front();
        DenseMap<Value, std::string> pyNames;
        unsigned counter = 0;
        SmallVector<std::string> inputStreams, outputStreams;

        py << "#!/usr/bin/env cs_python\n";
        py << "# Auto-generated by air-translate --emit-csl\n";
        py << "\n";
        py << "from cerebras.geometry.geometry import IntVector, IntRectangle\n";
        py << "from cerebras.sdk.runtime.sdkruntimepybind import (\n";
        py << "    Color, Edge, Route, RoutingPosition, get_edge_routing,\n";
        py << "    SdkLayout, SdkTarget, SdkRuntime, SimfabConfig, get_platform,\n";
        py << ")\n";
        py << "\n";
        py << "layout = SdkLayout()\n";
        py << "\n";

        return emitLayoutBody(py, body, kernelForRegion, pyNames, counter,
                              inputStreams, outputStreams);
      },
      diagOS);
}

//===----------------------------------------------------------------------===//
// emitRunPy — emit run.py (combined layout + compile + SdkRuntime)
//===----------------------------------------------------------------------===//

LogicalResult CSLTextEmitter::emitRunPy(
    csl::SpatialPlacementOp placement,
    DenseMap<Value, csl::KernelOp> &kernelForRegion,
    raw_ostream &diagOS) {

  Block &body = placement.getBody().front();

  // --- Collect exported variables from kernel comptime blocks ---
  struct ExportedSym {
    std::string name;
    std::string cslType; // e.g. "i16", "f32"
    int peX = 0, peY = 0;
  };
  SmallVector<ExportedSym> exportedSyms;
  llvm::DenseSet<csl::KernelOp> visited;
  for (auto &kv : kernelForRegion) {
    csl::KernelOp kernel = kv.second;
    if (!visited.insert(kernel).second)
      continue;
    kernel.getBody().walk([&](csl::ComptimeOp comptime) {
      for (auto &op : comptime.getBody().front()) {
        if (auto exSym = dyn_cast<csl::ExportSymbolOp>(op)) {
          std::string symName = exSym.getSym().str();
          std::string cslTy = "/* unknown */";
          kernel.getBody().walk([&](csl::VarOp varOp) {
            if (varOp.getSymName() == symName)
              cslTy = CSLWriter::mapType(varOp.getType());
          });
          exportedSyms.push_back({symName, cslTy, 0, 0});
        }
      }
    });
  }

  // --- Collect compile-time parameters (set_param_all) for header section ---
  struct SetParamInfo {
    std::string paramName;
    std::string value;
  };
  SmallVector<SetParamInfo> setParamAllInfos;
  for (auto &op : body) {
    if (auto setParam = dyn_cast<csl::SetParamAllOp>(op)) {
      std::string valStr;
      auto valAttr = setParam.getValue();
      if (auto intAttr = dyn_cast<IntegerAttr>(valAttr))
        valStr = std::to_string(intAttr.getValue().getSExtValue());
      else if (auto fltAttr = dyn_cast<FloatAttr>(valAttr))
        valStr = std::to_string(fltAttr.getValueAsDouble());
      else
        valStr = "# unsupported value";
      setParamAllInfos.push_back({setParam.getParamName().str(), valStr});
    }
  }

  return writePyToFile(
      "run.py",
      [&](raw_ostream &py) -> LogicalResult {
        py << "#!/usr/bin/env cs_python\n";
        py << "# Auto-generated by air-translate --emit-csl\n";
        py << "\n";
        py << "import argparse\n";
        py << "import numpy as np\n";
        py << "from cerebras.geometry.geometry import IntVector, IntRectangle\n";
        py << "from cerebras.sdk.runtime.sdkruntimepybind import (\n";
        py << "    Color, Edge, Route, RoutingPosition, get_edge_routing,\n";
        py << "    SdkLayout, SdkTarget, SdkRuntime, SimfabConfig, get_platform,\n";
        py << ")\n";
        py << "\n";
        py << "parser = argparse.ArgumentParser()\n";
        py << "parser.add_argument('--cmaddr', help='IP:port for CS system')\n";
        py << "parser.add_argument('--arch', choices=['wse2', 'wse3'], default='wse3',\n";
        py << "    help='Target WSE architecture (default: wse3)')\n";
        py << "args = parser.parse_args()\n";
        py << "\n";

        // Emit compile-time parameter constant declarations
        if (!setParamAllInfos.empty()) {
          py << "###########\n";
          py << "### Parameters\n";
          py << "###########\n";
          for (auto &sp : setParamAllInfos)
            py << sp.paramName << " = " << sp.value << "\n";
          py << "\n";
        }

        py << "###########\n";
        py << "### Layout\n";
        py << "###########\n";
        py << "config = SimfabConfig(dump_core=True)\n";
        py << "target = SdkTarget.WSE3 if (args.arch == 'wse3') else SdkTarget.WSE2\n";
        py << "platform = get_platform(args.cmaddr, config, target)\n";
        py << "layout = SdkLayout(platform)\n";
        py << "\n";

        // Emit layout body (single-pass)
        DenseMap<Value, std::string> pyNames;
        unsigned counter = 0;
        SmallVector<std::string> inputStreamNames, outputStreamNames;
        if (failed(emitLayoutBody(py, body, kernelForRegion, pyNames, counter,
                                  inputStreamNames, outputStreamNames)))
          return failure();

        py << "\n";
        py << "compile_artifacts = layout.compile(out_prefix='out')\n";
        py << "\n";

        py << "############\n";
        py << "### Runtime\n";
        py << "############\n";
        py << "runtime = SdkRuntime(compile_artifacts, platform, memcpy_required=False)\n";
        py << "runtime.load()\n";
        py << "runtime.run()\n";

        // Emit stream send/receive placeholders
        for (auto &sname : inputStreamNames)
          py << "# runtime.send(" << sname << ", data, nonblock=True)   # TODO: fill in data\n";
        for (auto &sname : outputStreamNames)
          py << "# runtime.receive(" << sname << ", result, size, nonblock=True)  # TODO: fill in result\n";

        py << "runtime.stop()\n";
        py << "\n";

        if (!exportedSyms.empty()) {
          py << "#################\n";
          py << "### Verification\n";
          py << "#################\n";
          for (auto &sym : exportedSyms) {
            std::string dtype;
            if (sym.cslType == "i16" || sym.cslType == "u16")
              dtype = "uint16";
            else if (sym.cslType == "i32" || sym.cslType == "u32")
              dtype = "uint32";
            else if (sym.cslType == "f32")
              dtype = "float32";
            else if (sym.cslType == "f16" || sym.cslType == "bf16")
              dtype = "float16";
            else
              dtype = "uint32";

            py << "result_" << sym.name
               << " = runtime.read_symbol(" << sym.peX << ", " << sym.peY
               << ", '" << sym.name << "', dtype='" << dtype << "')\n";
          }
          py << "\n";

          // Generate assertions for exported syms matched with set_param_all
          for (auto &sym : exportedSyms) {
            for (auto &sp : setParamAllInfos) {
              if (sp.paramName == sym.name) {
                py << "assert result_" << sym.name
                   << " == [" << sp.paramName << "], \""
                   << sym.name << " mismatch\"\n";
              }
            }
          }
        }

        py << "print(\"SUCCESS!\")\n";
        return success();
      },
      diagOS);
}

//===----------------------------------------------------------------------===//
// emitSpatialPlacement — emit kernel .csl files + layout.py + run.py
//===----------------------------------------------------------------------===//

LogicalResult CSLTextEmitter::emitSpatialPlacement(
    mlir::ModuleOp mlirModule, csl::SpatialPlacementOp placement,
    raw_ostream &diagOS) {

  Block &body = placement.getBody().front();

  // --- Build kernelForRegion map: region SSA value -> KernelOp ---
  // Walk all csl.place ops to find which kernel is placed on which region.
  // Also collect kernels defined directly in the body.
  DenseMap<Value, csl::KernelOp> kernelForRegion;
  DenseMap<Value, csl::KernelOp> allKernels;

  for (auto &op : body) {
    if (auto k = dyn_cast<csl::KernelOp>(op))
      allKernels[k.getResult()] = k;
    if (auto placeOp = dyn_cast<csl::PlaceOp>(op)) {
      if (auto k = placeOp.getKernel().getDefiningOp<csl::KernelOp>()) {
        allKernels[k.getResult()] = k;
        kernelForRegion[placeOp.getRegion()] = k;
      }
    }
  }

  // --- Emit one .csl file per csl.kernel ---
  llvm::DenseSet<csl::KernelOp> emittedKernels;
  for (auto &kv : allKernels) {
    if (!emittedKernels.insert(kv.second).second)
      continue;
    if (failed(emitKernelFile(kv.second, diagOS)))
      return failure();
  }

  // --- Emit layout.py ---
  if (failed(emitLayoutPy(placement, kernelForRegion, diagOS)))
    return failure();

  // --- Emit run.py (host-side runtime) ---
  return emitRunPy(placement, kernelForRegion, diagOS);
}

//===----------------------------------------------------------------------===//
// translate — entry point
//===----------------------------------------------------------------------===//

LogicalResult CSLTextEmitter::translate(mlir::ModuleOp module,
                                         raw_ostream &diagOS) {
  // Create output directory
  if (auto ec = llvm::sys::fs::create_directories(outputDir)) {
    llvm::errs() << "Cannot create output directory '" << outputDir
                 << "': " << ec.message() << "\n";
    return failure();
  }

  bool emittedSomething = false;

  for (auto &op : module.getOps()) {
    if (auto modOp = dyn_cast<csl::ModuleOp>(op)) {
      if (failed(emitModule(module, modOp, diagOS)))
        return failure();
      emittedSomething = true;
    } else if (auto placement = dyn_cast<csl::SpatialPlacementOp>(op)) {
      if (failed(emitSpatialPlacement(module, placement, diagOS)))
        return failure();
      emittedSomething = true;
    }
    // Kernels at top-level (outside spatial_placement) also need to be emitted.
    else if (auto kernel = dyn_cast<csl::KernelOp>(op)) {
      if (failed(emitKernelFile(kernel, diagOS)))
        return failure();
      emittedSomething = true;
    }
    // func.func and other non-CSL ops at module level are silently skipped
  }

  if (!emittedSomething)
    diagOS << "Warning: No csl.module or csl.spatial_placement found in "
              "input; nothing emitted.\n";

  return success();
}

} // anonymous namespace

//===----------------------------------------------------------------------===//
// Translation registration
//===----------------------------------------------------------------------===//

static llvm::cl::opt<std::string> clCSLOutputDir(
    "csl-output-dir",
    llvm::cl::desc("Output directory for --emit-csl generated CSL files"),
    llvm::cl::init("."));

void registerCSLToTextTranslation() {
  TranslateFromMLIRRegistration reg(
      "emit-csl", "Emit CSL dialect IR to Cerebras CSL/Python files",
      [](mlir::ModuleOp module, raw_ostream &output) {
        CSLTextEmitter emitter(clCSLOutputDir);
        return emitter.translate(module, output);
      },
      [](DialectRegistry &registry) {
        registry.insert<xilinx::csl::CSLDialect, func::FuncDialect,
                        arith::ArithDialect, memref::MemRefDialect,
                        scf::SCFDialect>();
      });
}

} // namespace csl
} // namespace xilinx
