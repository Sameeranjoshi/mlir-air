//===- AIRToCSLPass.cpp -----------------------------------------*- C++ -*-===//
//
// This pass lowers AIR dialect operations (air.launch, air.segment, air.herd)
// to Cerebras CSL text files: layout.csl, pe_program.csl, and run.py.
//
// NOTE: This pass is currently kept for reference and later use. Our modern
// pipeline should lower AIR to the newly refined CSL MLIR Dialect 
// (csl.spatial_placement, csl.code_region, etc.), and from there an MLIR 
// Translation pass can generate the CSL native code.
//
//===----------------------------------------------------------------------===//

#include "air/Conversion/PassDetail.h"

#include "air/Dialect/AIR/AIRDialect.h"
#include "air/Util/Util.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

#include <string>

#define DEBUG_TYPE "air-to-csl"

using namespace mlir;

namespace xilinx {
namespace air {

// Map MLIR element types to CSL type names.
static StringRef getCSLTypeName(Type ty) {
  if (ty.isF32())
    return "f32";
  if (ty.isF16())
    return "f16";
  if (ty.isInteger(32))
    return "i32";
  if (ty.isInteger(16))
    return "i16";
  if (ty.isInteger(8))
    return "i8";
  if (ty.isIndex())
    return "i32";
  return "f32";
}

// Compute a flat size from a ranked memref shape.
static int64_t getFlatSize(MemRefType memTy) {
  int64_t total = 1;
  for (int64_t dim : memTy.getShape()) {
    if (ShapedType::isDynamic(dim))
      return -1;
    total *= dim;
  }
  return total;
}

// Return the bit-width string for memcpy data type selection in run.py.
static std::string getMemcpyDataTypeStr(Type elemTy) {
  unsigned bits = elemTy.isIndex() ? 32 : elemTy.getIntOrFloatBitWidth();
  return "MEMCPY_" + std::to_string(bits) + "BIT";
}

//===----------------------------------------------------------------------===//
// CSLEmitter -- walks AIR IR and emits CSL text
//===----------------------------------------------------------------------===//

class CSLEmitter {
public:
  CSLEmitter(StringRef outputDir) : outputDir(outputDir) {}

  LogicalResult emit(ModuleOp module);

private:
  std::string outputDir;

  struct HerdInfo {
    air::HerdOp op;
    uint64_t numCols = 1;
    uint64_t numRows = 1;
    std::string name;
  };

  struct SegmentInfo {
    air::SegmentOp op;
    uint64_t width = 1;
    uint64_t height = 1;
    std::string name;
    SmallVector<HerdInfo> herds;
  };

  struct ExportedArray {
    std::string name;
    std::string cslType; // e.g. "[*]f32"
    bool mutable_ = false;
    int64_t numElements = 0;
    Type elemType;
  };

  // Emit the three output files.
  LogicalResult emitLayoutCSL(raw_ostream &os, const SegmentInfo &seg,
                              ArrayRef<ExportedArray> exports);
  LogicalResult emitPEProgramCSL(raw_ostream &os, HerdInfo &herd,
                                 ArrayRef<ExportedArray> exports);
  LogicalResult emitRunPy(raw_ostream &os, const SegmentInfo &seg,
                          ArrayRef<ExportedArray> exports);

  // Walk the herd body and emit CSL statements.
  void emitHerdBody(raw_ostream &os, air::HerdOp herd,
                    DenseMap<Value, std::string> &valueNames, unsigned indent);

  // Emit a single MLIR operation as CSL code.
  void emitOp(raw_ostream &os, Operation *op,
              DenseMap<Value, std::string> &valueNames, unsigned indent);

  // Generate a fresh variable name.
  std::string freshName(StringRef prefix = "v") {
    return (prefix + Twine(nameCounter++)).str();
  }

  unsigned nameCounter = 0;

  // Indent helper.
  static std::string ind(unsigned level) { return std::string(level * 2, ' '); }
};

LogicalResult CSLEmitter::emit(ModuleOp module) {
  SmallVector<SegmentInfo> segments;
  SmallVector<ExportedArray> allExports;

  // Walk the module to collect structure.
  module.walk([&](air::LaunchOp launch) {
    launch.walk([&](air::SegmentOp segment) {
      SegmentInfo seg;
      seg.op = segment;
      seg.name = segment.getSymName().value_or("segment_0").str();

      segment.walk([&](air::HerdOp herd) {
        HerdInfo hi;
        hi.op = herd;
        hi.numCols = herd.getNumCols();
        hi.numRows = herd.getNumRows();
        hi.name = herd.getSymName().value_or("herd_0").str();
        seg.herds.push_back(hi);
      });

      // Segment dimensions: use herd dimensions if available.
      if (!seg.herds.empty()) {
        seg.width = seg.herds[0].numCols;
        seg.height = seg.herds[0].numRows;
      }
      if (auto cols = segment.getNumCols())
        seg.width = *cols;
      if (auto rows = segment.getNumRows())
        seg.height = *rows;

      segments.push_back(seg);
    });
  });

  // Also handle bare herds (no launch/segment wrapper) as a single segment.
  if (segments.empty()) {
    module.walk([&](air::HerdOp herd) {
      // Skip herds already inside a segment.
      if (herd->getParentOfType<air::SegmentOp>())
        return;
      SegmentInfo seg;
      seg.name = "segment_0";
      HerdInfo hi;
      hi.op = herd;
      hi.numCols = herd.getNumCols();
      hi.numRows = herd.getNumRows();
      hi.name = herd.getSymName().value_or("herd_0").str();
      seg.herds.push_back(hi);
      seg.width = hi.numCols;
      seg.height = hi.numRows;
      segments.push_back(seg);
    });
  }

  if (segments.empty()) {
    module.emitWarning("No air.segment or air.herd found; nothing to emit.");
    return success();
  }

  // Collect exported arrays from herd kernel arguments (memref operands).
  for (auto &seg : segments) {
    for (auto &herd : seg.herds) {
      auto kernelArgs = herd.op.getKernelArguments();
      for (unsigned i = 0; i < kernelArgs.size(); ++i) {
        auto argTy = kernelArgs[i].getType();
        if (auto memTy = dyn_cast<MemRefType>(argTy)) {
          ExportedArray ea;
          ea.name = "arg_" + std::to_string(i);
          ea.elemType = memTy.getElementType();
          ea.cslType = "[*]" + getCSLTypeName(memTy.getElementType()).str();
          ea.numElements = getFlatSize(memTy);
          ea.mutable_ = true;
          allExports.push_back(ea);
        }
      }
    }
  }

  auto &seg = segments[0];

  // Write layout.csl
  {
    SmallString<256> path(outputDir);
    llvm::sys::path::append(path, "layout.csl");
    std::error_code ec;
    llvm::raw_fd_ostream file(path, ec);
    if (ec) {
      module.emitError("Failed to open ") << path << ": " << ec.message();
      return failure();
    }
    if (failed(emitLayoutCSL(file, seg, allExports)))
      return failure();
  }

  // Write pe_program.csl (from first herd)
  if (!seg.herds.empty()) {
    SmallString<256> path(outputDir);
    llvm::sys::path::append(path, "pe_program.csl");
    std::error_code ec;
    llvm::raw_fd_ostream file(path, ec);
    if (ec) {
      module.emitError("Failed to open ") << path << ": " << ec.message();
      return failure();
    }
    if (failed(emitPEProgramCSL(file, seg.herds[0], allExports)))
      return failure();
  }

  // Write run.py
  {
    SmallString<256> path(outputDir);
    llvm::sys::path::append(path, "run.py");
    std::error_code ec;
    llvm::raw_fd_ostream file(path, ec);
    if (ec) {
      module.emitError("Failed to open ") << path << ": " << ec.message();
      return failure();
    }
    if (failed(emitRunPy(file, seg, allExports)))
      return failure();
  }

  return success();
}

//===----------------------------------------------------------------------===//
// layout.csl emission
//===----------------------------------------------------------------------===//

LogicalResult CSLEmitter::emitLayoutCSL(raw_ostream &os,
                                        const SegmentInfo &seg,
                                        ArrayRef<ExportedArray> exports) {
  uint64_t W = seg.width;
  uint64_t H = seg.height;

  os << "// Auto-generated by air-to-csl pass\n\n";
  os << "const memcpy = @import_module(\"<memcpy/get_params>\", "
     << ".{ .width = " << W << ", .height = " << H << " });\n\n";

  os << "layout {\n";
  os << "  @set_rectangle(" << W << ", " << H << ");\n\n";

  for (uint64_t row = 0; row < H; ++row) {
    for (uint64_t col = 0; col < W; ++col) {
      os << "  @set_tile_code(" << col << ", " << row
         << ", \"pe_program.csl\", .{ .memcpy_params = memcpy.get_params("
         << col << ") });\n";
    }
  }

  os << "\n";

  // Export symbols for host-accessible arrays.
  for (auto &exp : exports) {
    os << "  @export_name(\"" << exp.name << "\", " << exp.cslType << ", "
       << (exp.mutable_ ? "true" : "false") << ");\n";
  }

  // Export the main compute function.
  os << "  @export_name(\"init_and_compute\", fn()void);\n";

  os << "}\n";
  return success();
}

//===----------------------------------------------------------------------===//
// pe_program.csl emission
//===----------------------------------------------------------------------===//

LogicalResult CSLEmitter::emitPEProgramCSL(raw_ostream &os,
                                           HerdInfo &herd,
                                           ArrayRef<ExportedArray> exports) {
  os << "// Auto-generated by air-to-csl pass\n\n";
  os << "param memcpy_params: comptime_struct;\n";
  os << "const sys_mod = @import_module(\"<memcpy/memcpy>\", memcpy_params);\n\n";

  // Emit global arrays from kernel arguments (memref operands).
  auto kernelArgs = herd.op.getKernelArguments();
  DenseMap<Value, std::string> valueNames;

  unsigned argIdx = 0;
  for (auto arg : kernelArgs) {
    auto argTy = arg.getType();
    if (auto memTy = dyn_cast<MemRefType>(argTy)) {
      std::string name = "arg_" + std::to_string(argIdx);
      int64_t flatSize = getFlatSize(memTy);
      StringRef elemTypeName = getCSLTypeName(memTy.getElementType());
      if (flatSize > 0) {
        os << "var " << name << ": [" << flatSize << "]" << elemTypeName
           << ";\n";
      } else {
        os << "// TODO: dynamic-sized memref arg_" << argIdx << "\n";
      }
      valueNames[arg] = name;
    } else {
      std::string name = "arg_" + std::to_string(argIdx);
      os << "const " << name << ": " << getCSLTypeName(argTy)
         << " = undefined; // scalar kernel arg\n";
      valueNames[arg] = name;
    }
    argIdx++;
  }

  // Emit pointer exports for host-accessible arrays.
  for (auto &exp : exports) {
    os << "const " << exp.name << "_ptr: " << exp.cslType << " = &"
       << exp.name << ";\n";
  }
  os << "\n";

  // Walk the herd body to discover local allocs and computation.
  // First pass: collect local allocs.
  Block &body = herd.op.getBody().front();
  SmallVector<memref::AllocOp> localAllocs;
  for (auto &op : body) {
    if (auto alloc = dyn_cast<memref::AllocOp>(op)) {
      auto memTy = alloc.getType();
      std::string name = freshName("buf_");
      int64_t flatSize = getFlatSize(memTy);
      StringRef elemTypeName = getCSLTypeName(memTy.getElementType());
      if (flatSize > 0) {
        os << "var " << name << ": [" << flatSize << "]" << elemTypeName
           << ";\n";
      } else {
        os << "// TODO: dynamic alloc " << name << "\n";
      }
      valueNames[alloc.getResult()] = name;
      localAllocs.push_back(alloc);
    }
  }
  os << "\n";

  // Emit the compute function containing the herd body logic.
  os << "fn compute() void {\n";
  emitHerdBody(os, herd.op, valueNames, 1);
  os << "}\n\n";

  // Emit the init_and_compute wrapper.
  os << "fn init_and_compute() void {\n";
  os << "  compute();\n";
  os << "  sys_mod.unblock_cmd_stream();\n";
  os << "}\n\n";

  // Emit comptime block with exports.
  os << "comptime {\n";
  for (auto &exp : exports) {
    os << "  @export_symbol(" << exp.name << "_ptr, \"" << exp.name
       << "\");\n";
  }
  os << "  @export_symbol(init_and_compute);\n";
  os << "}\n";

  return success();
}

//===----------------------------------------------------------------------===//
// Herd body -> CSL code emission
//===----------------------------------------------------------------------===//

void CSLEmitter::emitHerdBody(raw_ostream &os, air::HerdOp herd,
                              DenseMap<Value, std::string> &valueNames,
                              unsigned indent) {
  // Map the tile IDs to constants (since each PE has known coordinates).
  auto ids = herd.getIds();
  if (ids.size() >= 1)
    valueNames[ids[0]] = "0"; // tile x
  if (ids.size() >= 2)
    valueNames[ids[1]] = "0"; // tile y

  auto sizes = herd.getSize();
  if (sizes.size() >= 1)
    valueNames[sizes[0]] = std::to_string(herd.getNumCols());
  if (sizes.size() >= 2)
    valueNames[sizes[1]] = std::to_string(herd.getNumRows());

  Block &body = herd.getBody().front();
  for (auto &op : body) {
    // Skip allocs (already emitted as globals) and terminators.
    if (isa<memref::AllocOp>(op) || isa<memref::DeallocOp>(op))
      continue;
    if (isa<air::HerdTerminatorOp>(op))
      continue;
    emitOp(os, &op, valueNames, indent);
  }
}

void CSLEmitter::emitOp(raw_ostream &os, Operation *op,
                        DenseMap<Value, std::string> &valueNames,
                        unsigned indent) {
  auto getVal = [&](Value v) -> std::string {
    auto it = valueNames.find(v);
    if (it != valueNames.end())
      return it->second;
    return "/* unknown */";
  };

  // arith.constant
  if (auto cst = dyn_cast<arith::ConstantOp>(op)) {
    std::string name = freshName("c_");
    auto attr = cst.getValue();
    if (auto intAttr = dyn_cast<IntegerAttr>(attr)) {
      os << ind(indent) << "const " << name << ": "
         << getCSLTypeName(cst.getType()) << " = " << intAttr.getInt()
         << ";\n";
    } else if (auto fpAttr = dyn_cast<FloatAttr>(attr)) {
      SmallString<32> fpStr;
      fpAttr.getValue().toString(fpStr, /*FormatPrecision=*/6);
      os << ind(indent) << "const " << name << ": "
         << getCSLTypeName(cst.getType()) << " = " << fpStr << ";\n";
    } else {
      os << ind(indent) << "const " << name << " = undefined; "
         << "// TODO: unsupported constant type\n";
    }
    valueNames[cst.getResult()] = name;
    return;
  }

  // arith binary ops
  if (auto addF = dyn_cast<arith::AddFOp>(op)) {
    std::string name = freshName("v_");
    os << ind(indent) << "var " << name << ": "
       << getCSLTypeName(addF.getType()) << " = " << getVal(addF.getLhs())
       << " + " << getVal(addF.getRhs()) << ";\n";
    valueNames[addF.getResult()] = name;
    return;
  }
  if (auto mulF = dyn_cast<arith::MulFOp>(op)) {
    std::string name = freshName("v_");
    os << ind(indent) << "var " << name << ": "
       << getCSLTypeName(mulF.getType()) << " = " << getVal(mulF.getLhs())
       << " * " << getVal(mulF.getRhs()) << ";\n";
    valueNames[mulF.getResult()] = name;
    return;
  }
  if (auto addI = dyn_cast<arith::AddIOp>(op)) {
    std::string name = freshName("v_");
    os << ind(indent) << "var " << name << ": "
       << getCSLTypeName(addI.getType()) << " = " << getVal(addI.getLhs())
       << " + " << getVal(addI.getRhs()) << ";\n";
    valueNames[addI.getResult()] = name;
    return;
  }
  if (auto mulI = dyn_cast<arith::MulIOp>(op)) {
    std::string name = freshName("v_");
    os << ind(indent) << "var " << name << ": "
       << getCSLTypeName(mulI.getType()) << " = " << getVal(mulI.getLhs())
       << " * " << getVal(mulI.getRhs()) << ";\n";
    valueNames[mulI.getResult()] = name;
    return;
  }
  if (auto subF = dyn_cast<arith::SubFOp>(op)) {
    std::string name = freshName("v_");
    os << ind(indent) << "var " << name << ": "
       << getCSLTypeName(subF.getType()) << " = " << getVal(subF.getLhs())
       << " - " << getVal(subF.getRhs()) << ";\n";
    valueNames[subF.getResult()] = name;
    return;
  }
  if (auto subI = dyn_cast<arith::SubIOp>(op)) {
    std::string name = freshName("v_");
    os << ind(indent) << "var " << name << ": "
       << getCSLTypeName(subI.getType()) << " = " << getVal(subI.getLhs())
       << " - " << getVal(subI.getRhs()) << ";\n";
    valueNames[subI.getResult()] = name;
    return;
  }

  // memref.load
  if (auto load = dyn_cast<memref::LoadOp>(op)) {
    std::string name = freshName("ld_");
    std::string memName = getVal(load.getMemRef());
    // Build index expression.
    SmallVector<std::string> idxStrs;
    for (auto idx : load.getIndices())
      idxStrs.push_back(getVal(idx));

    os << ind(indent) << "const " << name << ": "
       << getCSLTypeName(load.getType()) << " = " << memName << "[";
    if (idxStrs.size() == 1) {
      os << idxStrs[0];
    } else if (idxStrs.empty()) {
      os << "0";
    } else {
      // Multi-dimensional: linearize using the memref's shape.
      auto memTy = load.getMemRefType();
      auto shape = memTy.getShape();
      // row-major linearization
      bool first = true;
      for (unsigned d = 0; d < idxStrs.size(); ++d) {
        if (!first)
          os << " + ";
        os << idxStrs[d];
        for (unsigned k = d + 1; k < shape.size(); ++k)
          os << "*" << shape[k];
        first = false;
      }
    }
    os << "];\n";
    valueNames[load.getResult()] = name;
    return;
  }

  // memref.store
  if (auto store = dyn_cast<memref::StoreOp>(op)) {
    std::string memName = getVal(store.getMemRef());
    std::string valName = getVal(store.getValueToStore());
    SmallVector<std::string> idxStrs;
    for (auto idx : store.getIndices())
      idxStrs.push_back(getVal(idx));

    os << ind(indent) << memName << "[";
    if (idxStrs.size() == 1) {
      os << idxStrs[0];
    } else if (idxStrs.empty()) {
      os << "0";
    } else {
      auto memTy = store.getMemRefType();
      auto shape = memTy.getShape();
      bool first = true;
      for (unsigned d = 0; d < idxStrs.size(); ++d) {
        if (!first)
          os << " + ";
        os << idxStrs[d];
        for (unsigned k = d + 1; k < shape.size(); ++k)
          os << "*" << shape[k];
        first = false;
      }
    }
    os << "] = " << valName << ";\n";
    return;
  }

  // scf.for -> CSL for with @range or while
  if (auto forOp = dyn_cast<scf::ForOp>(op)) {
    std::string iv = freshName("i_");
    valueNames[forOp.getInductionVar()] = iv;

    std::string lb = getVal(forOp.getLowerBound());
    std::string ub = getVal(forOp.getUpperBound());

    os << ind(indent) << "var " << iv << ": i32 = " << lb << ";\n";
    os << ind(indent) << "while (" << iv << " < " << ub << ") : (" << iv
       << " += 1) {\n";

    for (auto &bodyOp : forOp.getBody()->getOperations()) {
      if (isa<scf::YieldOp>(bodyOp))
        continue;
      emitOp(os, &bodyOp, valueNames, indent + 1);
    }

    os << ind(indent) << "}\n";
    return;
  }

  // air.dma_memcpy_nd -- emit as a comment annotation
  if (isa<air::DmaMemcpyNdOp>(op)) {
    os << ind(indent) << "// TODO: air.dma_memcpy_nd -> data movement\n";
    return;
  }

  // air.execute -- inline the body
  if (auto exec = dyn_cast<air::ExecuteOp>(op)) {
    for (auto &bodyOp : exec.getBody()) {
      if (isa<air::ExecuteTerminatorOp>(bodyOp))
        continue;
      emitOp(os, &bodyOp, valueNames, indent);
    }
    return;
  }

  // air.wait_all -- no-op in CSL
  if (isa<air::WaitAllOp>(op))
    return;

  // func.call
  if (auto call = dyn_cast<func::CallOp>(op)) {
    os << ind(indent) << call.getCallee() << "(";
    bool first = true;
    for (auto arg : call.getOperands()) {
      if (!first)
        os << ", ";
      os << getVal(arg);
      first = false;
    }
    os << ");\n";
    return;
  }

  // arith.index_cast
  if (auto cast = dyn_cast<arith::IndexCastOp>(op)) {
    std::string name = freshName("v_");
    os << ind(indent) << "const " << name << " = @as("
       << getCSLTypeName(cast.getType()) << ", " << getVal(cast.getIn())
       << ");\n";
    valueNames[cast.getResult()] = name;
    return;
  }

  // Fallback: emit as a TODO comment.
  os << ind(indent) << "// TODO: unsupported op: " << op->getName() << "\n";
}

//===----------------------------------------------------------------------===//
// run.py emission
//===----------------------------------------------------------------------===//

LogicalResult CSLEmitter::emitRunPy(raw_ostream &os, const SegmentInfo &seg,
                                    ArrayRef<ExportedArray> exports) {
  uint64_t W = seg.width;
  uint64_t H = seg.height;

  os << "#!/usr/bin/env cs_python\n";
  os << "# Auto-generated by air-to-csl pass\n\n";
  os << "import argparse\n";
  os << "import numpy as np\n\n";
  os << "from cerebras.sdk.runtime.sdkruntimepybind import "
     << "SdkRuntime, MemcpyDataType, MemcpyOrder\n\n";

  os << "parser = argparse.ArgumentParser()\n";
  os << "parser.add_argument('--name', help='the test compile output dir')\n";
  os << "parser.add_argument('--cmaddr', help='IP:port for CS system')\n";
  os << "args = parser.parse_args()\n\n";

  os << "runner = SdkRuntime(args.name, cmaddr=args.cmaddr)\n\n";

  // Get symbols for exported arrays.
  for (auto &exp : exports) {
    os << exp.name << "_symbol = runner.get_id('" << exp.name << "')\n";
  }
  os << "\n";

  os << "runner.load()\n";
  os << "runner.run()\n\n";

  // Copy input data to device (H2D) for mutable exports.
  for (auto &exp : exports) {
    if (exp.mutable_ && exp.numElements > 0) {
      std::string npDtype =
          exp.elemType.isF32()  ? "np.float32"
          : exp.elemType.isF16() ? "np.float16"
                                 : "np.int32";
      os << "# TODO: initialize " << exp.name
         << " with actual data and transfer H2D\n";
      os << "# " << exp.name << "_data = np.zeros(" << exp.numElements
         << ", dtype=" << npDtype << ")\n";
      os << "# runner.memcpy_h2d(" << exp.name << "_symbol, " << exp.name
         << "_data, 0, 0, " << W << ", " << H << ", " << exp.numElements
         << ",\n";
      os << "#   streaming=False, order=MemcpyOrder.ROW_MAJOR,\n";
      os << "#   data_type=MemcpyDataType." << getMemcpyDataTypeStr(exp.elemType)
         << ", nonblock=False)\n\n";
    }
  }

  os << "runner.launch('init_and_compute', nonblock=False)\n\n";

  // Copy results back from device (D2H).
  for (auto &exp : exports) {
    if (exp.numElements > 0) {
      std::string npDtype =
          exp.elemType.isF32()  ? "np.float32"
          : exp.elemType.isF16() ? "np.float16"
                                 : "np.int32";
      os << exp.name << "_result = np.zeros([" << W << "*" << H << "*"
         << exp.numElements << "], dtype=" << npDtype << ")\n";
      os << "runner.memcpy_d2h(" << exp.name << "_result, " << exp.name
         << "_symbol, 0, 0, " << W << ", " << H << ", " << exp.numElements
         << ",\n";
      os << "  streaming=False, order=MemcpyOrder.ROW_MAJOR,\n";
      os << "  data_type=MemcpyDataType."
         << getMemcpyDataTypeStr(exp.elemType)
         << ", nonblock=False)\n\n";
    }
  }

  os << "runner.stop()\n\n";
  os << "print('SUCCESS!')\n";

  return success();
}

//===----------------------------------------------------------------------===//
// Pass definition
//===----------------------------------------------------------------------===//

class AIRToCSLPass : public air::impl::AIRToCSLBase<AIRToCSLPass> {
public:
  AIRToCSLPass() = default;
  AIRToCSLPass(const AIRToCSLPass &pass) : AIRToCSLBase(pass) {}

  void runOnOperation() override {
    auto module = getOperation();

    if (auto ec = llvm::sys::fs::create_directories(clOutputDir)) {
      module.emitError("Cannot create output directory '")
          << clOutputDir << "': " << ec.message();
      return signalPassFailure();
    }

    CSLEmitter emitter(clOutputDir);
    if (failed(emitter.emit(module)))
      signalPassFailure();
  }
};

std::unique_ptr<mlir::Pass> createAIRToCSLPass() {
  return std::make_unique<AIRToCSLPass>();
}

} // namespace air
} // namespace xilinx
