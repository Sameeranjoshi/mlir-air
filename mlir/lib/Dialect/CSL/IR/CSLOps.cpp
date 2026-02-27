//===- CSLOps.cpp - CSL dialect operation implementation -------*- C++ -*-===//
//
// Part of the air-to-csl project.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#include "air/Dialect/CSL/CSLOps.h"
#include "air/Dialect/CSL/CSLDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"
#include "air/Dialect/CSL/CSLEnums.cpp.inc"
using namespace mlir;

//===----------------------------------------------------------------------===//
// csl.func
//===----------------------------------------------------------------------===//

void xilinx::csl::FuncOp::print(OpAsmPrinter &p) {
  p << ' ';
  p.printSymbolName(getSymName());
  p << "()";
  p << ' ';
  p.printRegion(getBody(), /*printEntryBlockArgs=*/false,
                /*printBlockTerminators=*/true);
}

ParseResult xilinx::csl::FuncOp::parse(OpAsmParser &parser,
                                        OperationState &result) {
  StringAttr nameAttr;
  if (parser.parseSymbolName(nameAttr, SymbolTable::getSymbolAttrName(),
                             result.attributes))
    return failure();

  if (parser.parseLParen() || parser.parseRParen())
    return failure();

  Region *body = result.addRegion();
  if (parser.parseRegion(*body, {}))
    return failure();

  if (body->empty())
    body->emplaceBlock();

  return success();
}

//===----------------------------------------------------------------------===//
// csl.task
//===----------------------------------------------------------------------===//

void xilinx::csl::TaskOp::print(OpAsmPrinter &p) {
  p << ' ';
  p.printSymbolName(getSymName());
  p << "()";
  p << " color(" << getColorId() << ")";
  p << ' ';
  p.printRegion(getBody(), /*printEntryBlockArgs=*/false,
                /*printBlockTerminators=*/true);
}

ParseResult xilinx::csl::TaskOp::parse(OpAsmParser &parser,
                                        OperationState &result) {
  StringAttr nameAttr;
  if (parser.parseSymbolName(nameAttr, SymbolTable::getSymbolAttrName(),
                             result.attributes))
    return failure();

  if (parser.parseLParen() || parser.parseRParen())
    return failure();

  if (parser.parseKeyword("color") || parser.parseLParen())
    return failure();

  int32_t colorId;
  if (parser.parseInteger(colorId))
    return failure();
  result.addAttribute("color_id",
                       parser.getBuilder().getI32IntegerAttr(colorId));

  if (parser.parseRParen())
    return failure();

  Region *body = result.addRegion();
  if (parser.parseRegion(*body, {}))
    return failure();

  if (body->empty())
    body->emplaceBlock();

  return success();
}

//===----------------------------------------------------------------------===//
// TableGen-generated op definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "air/Dialect/CSL/CSLOps.cpp.inc"
