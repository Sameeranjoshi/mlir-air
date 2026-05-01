//===- CSLOps.cpp - CSL dialect operation implementation -------*- C++ -*-===//
//
// Part of the air-to-csl project.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#include "air/Dialect/CSL/CSLOps.h"
#include "air/Dialect/CSL/CSLDialect.h"
#include "air/Dialect/CSL/CSLLayoutOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"
#include "air/Dialect/CSL/CSLEnums.cpp.inc"
using namespace mlir;

//===----------------------------------------------------------------------===//
// TableGen-generated op definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "air/Dialect/CSL/CSLOps.cpp.inc"

//===----------------------------------------------------------------------===//
// CSL_VarOp
//===----------------------------------------------------------------------===//

void xilinx::csl::VarOp::getAsmResultNames(
    OpAsmSetValueNameFn setNameFn) {
  setNameFn(getResult(), getSymName());
}

//===----------------------------------------------------------------------===//
// CSL v2: WaferOp — custom assembly: @sym_name {arch = "..."} { body }
//===----------------------------------------------------------------------===//

mlir::ParseResult xilinx::csl::WaferOp::parse(
    mlir::OpAsmParser &parser, mlir::OperationState &result) {
  // @sym_name
  StringAttr nameAttr;
  if (parser.parseSymbolName(nameAttr, mlir::SymbolTable::getSymbolAttrName(),
                              result.attributes))
    return mlir::failure();

  // Optional {arch = "...", ...} attribute dict
  if (parser.parseOptionalAttrDict(result.attributes))
    return mlir::failure();

  auto *body = result.addRegion();
  if (parser.parseRegion(*body, {}))
    return mlir::failure();
  if (body->empty())
    body->emplaceBlock();
  return mlir::success();
}

void xilinx::csl::WaferOp::print(mlir::OpAsmPrinter &printer) {
  printer << ' ';
  printer.printSymbolName(getSymName());
  printer.printOptionalAttrDict((*this)->getAttrs(),
      {WaferOp::getSymNameAttrName()});
  printer << ' ';
  printer.printRegion(getBody());
}

//===----------------------------------------------------------------------===//
// CSL v2: LayoutOp — custom assembly:
// {width = N : i64, height = M : i64} @sym_name { body }
//===----------------------------------------------------------------------===//

mlir::ParseResult xilinx::csl::LayoutOp::parse(
    mlir::OpAsmParser &parser, mlir::OperationState &result) {
  // {width = N, height = M, ...} attribute dict (comes before sym_name)
  if (parser.parseOptionalAttrDict(result.attributes))
    return mlir::failure();

  // @sym_name
  StringAttr nameAttr;
  if (parser.parseSymbolName(nameAttr, mlir::SymbolTable::getSymbolAttrName(),
                              result.attributes))
    return mlir::failure();

  auto *body = result.addRegion();
  if (parser.parseRegion(*body, {}))
    return mlir::failure();
  if (body->empty())
    body->emplaceBlock();
  return mlir::success();
}

void xilinx::csl::LayoutOp::print(mlir::OpAsmPrinter &printer) {
  printer.printOptionalAttrDict((*this)->getAttrs(),
      {LayoutOp::getSymNameAttrName()});
  printer << ' ';
  printer.printSymbolName(getSymName());
  printer << ' ';
  printer.printRegion(getBody());
}

//===----------------------------------------------------------------------===//
// CSL v2: CSLProgramOp — custom assembly for comptime block args
//
// Format: @sym_name (%arg: !csl.comptime<T>, ...) { body }
//         @sym_name { body }   (no-arg variant)
//===----------------------------------------------------------------------===//

mlir::ParseResult xilinx::csl::ProgramOp::parse(
    mlir::OpAsmParser &parser, mlir::OperationState &result) {
  using namespace mlir;

  // Parse @sym_name
  StringAttr nameAttr;
  if (parser.parseSymbolName(nameAttr, mlir::SymbolTable::getSymbolAttrName(),
                              result.attributes))
    return failure();

  // Parse optional ( %arg: !csl.comptime<T>, ... )
  SmallVector<OpAsmParser::Argument> regionArgs;
  if (succeeded(parser.parseOptionalLParen())) {
    // Empty parens: ()
    if (failed(parser.parseOptionalRParen())) {
      do {
        OpAsmParser::Argument arg;
        if (parser.parseArgument(arg, /*allowType=*/true))
          return failure();
        regionArgs.push_back(arg);
      } while (succeeded(parser.parseOptionalComma()));
      if (parser.parseRParen())
        return failure();
    }
  }

  // Capture the SSA names of the block arguments (without the leading '%')
  // as a `param_names` ArrayAttr so later passes and the printer can use
  // them for validation and round-trip preservation.
  if (!regionArgs.empty()) {
    SmallVector<Attribute> names;
    names.reserve(regionArgs.size());
    for (auto &arg : regionArgs) {
      StringRef rawName = arg.ssaName.name;
      if (rawName.starts_with("%"))
        rawName = rawName.drop_front();
      names.push_back(StringAttr::get(parser.getContext(), rawName));
    }
    result.addAttribute("param_names",
                        ArrayAttr::get(parser.getContext(), names));
  }

  auto *body = result.addRegion();
  if (parser.parseRegion(*body, regionArgs))
    return mlir::failure();
  if (body->empty())
    body->emplaceBlock();
  return mlir::success();
}

void xilinx::csl::ProgramOp::print(mlir::OpAsmPrinter &printer) {
  printer << ' ';
  printer.printSymbolName(getSymName());

  // Print block args if any, using the stored `param_names` when present
  // so round-trip preserves user-facing names like `%col` instead of `%arg0`.
  Block &entry = getBody().front();
  if (!entry.getArguments().empty()) {
    printer << '(';
    mlir::ArrayAttr names = getParamNamesAttr();
    for (unsigned i = 0, e = entry.getArguments().size(); i < e; ++i) {
      if (i > 0)
        printer << ", ";
      mlir::BlockArgument arg = entry.getArgument(i);
      if (names && i < names.size()) {
        printer << '%'
                << mlir::cast<mlir::StringAttr>(names[i]).getValue();
      } else {
        printer.printOperand(arg);
      }
      printer << ": ";
      printer.printType(arg.getType());
    }
    printer << ')';
  }

  // Suppress `param_names` from the attribute dict since it's printed
  // implicitly through the block-argument syntax above.
  printer.printOptionalAttrDict((*this)->getAttrs(),
      {ProgramOp::getSymNameAttrName(), "param_names"});

  printer << ' ';
  printer.printRegion(getBody(), /*printEntryBlockArgs=*/false);
}

//===----------------------------------------------------------------------===//
// CSL v2: HostOp — custom assembly for func-like args + layout attr
//
// Format: @sym_name (%arg: type, ...) {layout = @sym} { body }
//===----------------------------------------------------------------------===//

mlir::ParseResult xilinx::csl::HostOp::parse(
    mlir::OpAsmParser &parser, mlir::OperationState &result) {
  using namespace mlir;

  // Parse @sym_name
  StringAttr nameAttr;
  if (parser.parseSymbolName(nameAttr, mlir::SymbolTable::getSymbolAttrName(),
                              result.attributes))
    return failure();

  // Parse ( %arg: type, ... )
  SmallVector<OpAsmParser::Argument> args;
  if (parser.parseLParen())
    return failure();
  if (failed(parser.parseOptionalRParen())) {
    do {
      OpAsmParser::Argument arg;
      if (parser.parseArgument(arg, /*allowType=*/true))
        return failure();
      args.push_back(arg);
    } while (succeeded(parser.parseOptionalComma()));
    if (parser.parseRParen())
      return failure();
  }

  // Parse {layout = @sym}
  if (parser.parseLBrace())
    return failure();
  if (parser.parseKeyword("layout") || parser.parseEqual())
    return failure();
  FlatSymbolRefAttr layoutAttr;
  if (parser.parseAttribute(layoutAttr))
    return failure();
  result.addAttribute("layout", layoutAttr);
  if (parser.parseRBrace())
    return failure();

  auto *body = result.addRegion();
  if (parser.parseRegion(*body, args))
    return mlir::failure();
  if (body->empty())
    body->emplaceBlock();
  return mlir::success();
}

void xilinx::csl::HostOp::print(mlir::OpAsmPrinter &printer) {
  printer << ' ';
  printer.printSymbolName(getSymName());
  printer << '(';
  Block &entry = getBody().front();
  llvm::interleaveComma(entry.getArguments(), printer,
                        [&](mlir::BlockArgument arg) {
    printer.printOperand(arg);
    printer << ": ";
    printer.printType(arg.getType());
  });
  printer << ") {layout = ";
  printer.printAttributeWithoutType(getLayoutAttr());
  printer << "} ";
  printer.printOptionalAttrDict((*this)->getAttrs(),
      {HostOp::getSymNameAttrName(), "layout"});
  printer.printRegion(getBody(), /*printEntryBlockArgs=*/false);
}

//===----------------------------------------------------------------------===//
// csl.task — verifier: trigger_kind must match id/color attribute presence
//===----------------------------------------------------------------------===//

mlir::LogicalResult xilinx::csl::TaskOp::verify() {
  auto kind = getTriggerKind();
  bool hasId = (bool)getIdAttr();
  bool hasColor = (bool)getColorAttr();

  if (kind == "local_task_id") {
    if (!hasId)
      return emitOpError(
          "trigger_kind = \"local_task_id\" requires `id` attribute");
    if (hasColor)
      return emitOpError(
          "trigger_kind = \"local_task_id\" must not set `color`");
  } else if (kind == "color") {
    if (!hasColor)
      return emitOpError("trigger_kind = \"color\" requires `color` attribute");
    if (hasId)
      return emitOpError("trigger_kind = \"color\" must not set `id`");
  } else {
    return emitOpError("trigger_kind must be \"local_task_id\" or \"color\"")
           << " (got \"" << kind << "\")";
  }
  return mlir::success();
}

//===----------------------------------------------------------------------===//
// csl.get_fab_dsd — verifier: color symbol must resolve in csl.layout body
//===----------------------------------------------------------------------===//

mlir::LogicalResult xilinx::csl::GetFabDsdOp::verify() {
  // Walk up to the enclosing csl.wafer; csl.layout is its sibling of csl.program.
  auto wafer = (*this)->getParentOfType<xilinx::csl::WaferOp>();
  if (!wafer)
    return emitOpError("must be inside csl.wafer");

  // Find the csl.layout child by walking the wafer body once.
  xilinx::csl::LayoutOp layout;
  for (auto &op : wafer.getBody().front()) {
    if (auto l = mlir::dyn_cast<xilinx::csl::LayoutOp>(op)) {
      layout = l;
      break;
    }
  }
  if (!layout)
    return emitOpError("no csl.layout found in enclosing csl.wafer");

  auto color = mlir::SymbolTable::lookupSymbolIn(
      layout, getColorAttr().getAttr());
  if (!color)
    return emitOpError("references undefined color symbol '@")
           << getColorAttr().getValue() << "'";
  if (!mlir::isa<xilinx::csl::ColorOp>(color))
    return emitOpError("'@") << getColorAttr().getValue()
                              << "' is not a csl.color";
  return mlir::success();
}

//===----------------------------------------------------------------------===//
// csl.builtin_call — verifier: 'activate' requires 'async'
//===----------------------------------------------------------------------===//

mlir::LogicalResult xilinx::csl::BuiltinCallOp::verify() {
  if (getActivateAttr() && !getAsync())
    return emitOpError("'activate' requires 'async'");
  return mlir::success();
}

//===----------------------------------------------------------------------===//
// csl.stream.put — verifier
//===----------------------------------------------------------------------===//

mlir::LogicalResult xilinx::csl::StreamPutOp::verify() {
  // Source memref element type must be f32 this milestone.
  auto memTy = mlir::cast<mlir::MemRefType>(getSource().getType());
  if (!memTy.getElementType().isF32())
    return emitOpError("source memref element type must be f32 (got ")
           << memTy.getElementType() << ")";

  // Stream symbol must exist in enclosing csl.wafer's csl.layout.
  auto wafer = (*this)->getParentOfType<xilinx::csl::WaferOp>();
  if (!wafer)
    return emitOpError("must be inside csl.wafer");
  ::xilinx::csl_layout::StreamOp stream;
  for (auto &op : wafer.getBody().front()) {
    if (auto layout = mlir::dyn_cast<xilinx::csl::LayoutOp>(op)) {
      if (auto found = mlir::dyn_cast_or_null<::xilinx::csl_layout::StreamOp>(
              mlir::SymbolTable::lookupSymbolIn(layout,
                                                 getStreamAttr().getAttr())))
        stream = found;
      break;
    }
  }
  if (!stream)
    return emitOpError("references undefined stream '@")
           << getStreamAttr().getValue() << "'";
  return mlir::success();
}
