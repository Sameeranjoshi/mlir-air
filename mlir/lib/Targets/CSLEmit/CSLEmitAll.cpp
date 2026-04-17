//===- CSLEmitAll.cpp - Unified --emit-csl command -------------*- C++ -*-===//
//
// Part of the air-to-csl project.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
//
// Registers --emit-csl, which writes all the CSL output files into
// --output-dir=<path>:
//   <program>.csl       — device kernel
//   layout.csl          — cslc layout wrapper (wires memcpy_params)
//   csl_layout.py       — SdkLayout constructor (Python, optional alternative)
//   run.py              — runnable host driver (cs_python)
//   commands_wse3.sh    — one-command runner for CS-3 simulator
//
// The commands_wse3.sh script compiles `layout.csl` via `cslc --memcpy`
// into `compiled/` and then invoke `cs_python run.py --name=compiled`.
//
// Also provides registerCSLEmitTranslations() that registers all CSL
// translations in one call.
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
#include "mlir/Tools/mlir-translate/Translation.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

#include <memory>
#include <string>
#include <system_error>

using namespace mlir;

namespace xilinx {
namespace csl {

// Free-function entry points provided by the per-emitter TUs.
// Module-scoped variants (used by --emit-csl-program/layout/host registrations).
LogicalResult runProgramEmitter(ModuleOp module, llvm::raw_ostream &os);
LogicalResult runLayoutEmitter(ModuleOp module, llvm::raw_ostream &os);
LogicalResult runHostEmitter(ModuleOp module, llvm::raw_ostream &os);
// Wafer-scoped variants used by emitAll below.
LogicalResult runProgramEmitter(xilinx::csl::WaferOp wafer,
                                llvm::raw_ostream &os);
LogicalResult runLayoutEmitter(xilinx::csl::WaferOp wafer,
                               llvm::raw_ostream &os);
LogicalResult runHostEmitter(xilinx::csl::WaferOp wafer,
                             llvm::raw_ostream &os);

// Forward-declare the individual registration entry points.
void registerCSLProgramTranslation();
void registerCSLLayoutTranslation();
void registerCSLHostTranslation();

namespace {

static llvm::cl::opt<std::string> EmitCslOutputDir(
    "output-dir",
    llvm::cl::desc("Output directory for --emit-csl (required)"),
    llvm::cl::init(""));

/// Write `contents` to `<dir>/<filename>` and chmod it executable (0755).
static LogicalResult writeExecutable(ModuleOp module, llvm::StringRef dir,
                                     llvm::StringRef filename,
                                     llvm::StringRef contents) {
  llvm::SmallString<128> path(dir);
  llvm::sys::path::append(path, filename);
  std::error_code ec;
  {
    llvm::raw_fd_ostream out(llvm::StringRef(path.data(), path.size()), ec,
                             llvm::sys::fs::OF_Text);
    if (ec) {
      module.emitError() << "cannot open '" << path.c_str()
                         << "': " << ec.message();
      return failure();
    }
    out << contents;
  }
  // chmod 0755: owner rwx, group rx, world rx.
  namespace fs = llvm::sys::fs;
  ec = fs::setPermissions(
      llvm::StringRef(path.data(), path.size()),
      fs::owner_read | fs::owner_write | fs::owner_exe | fs::group_read |
          fs::group_exe | fs::all_read | fs::all_exe);
  if (ec) {
    module.emitError() << "cannot chmod '" << path.c_str()
                       << "': " << ec.message();
    return failure();
  }
  return success();
}

/// Map an MLIR element type to a CSL primitive type name.
static llvm::StringRef cslEltName(Type t) {
  if (t.isF32()) return "f32";
  if (t.isF16()) return "f16";
  if (t.isInteger(32)) return "i32";
  if (t.isInteger(16)) return "i16";
  return "f32";
}

struct ExportInfo {
  std::string alias;
  bool isFunc;
  std::string eltName;  // for var-kind exports: element type ("f32" etc.)
  bool writable;        // for var-kind exports: true if host h2d allowed
};

/// Walk csl.program ops to collect the (alias, type, direction) for each
/// host-visible export (var aliases + function exports).
static llvm::SmallVector<ExportInfo, 4>
collectExports(xilinx::csl::WaferOp wafer) {
  llvm::SmallVector<ExportInfo, 4> out;
  // Map var sym-name -> element type (within the first program).
  llvm::DenseMap<StringRef, Type> varElt;
  xilinx::csl::ProgramOp prog;
  wafer.walk([&](xilinx::csl::ProgramOp p) {
    if (!prog) prog = p;
  });
  if (!prog) return out;
  prog.walk([&](xilinx::csl::VarOp v) {
    if (auto memTy = dyn_cast<MemRefType>(v.getResult().getType()))
      varElt[v.getSymName()] = memTy.getElementType();
  });
  prog.walk([&](xilinx::csl::ExportOp e) {
    ExportInfo info;
    std::optional<StringRef> kind = e.getKind();
    info.isFunc = (kind.has_value() && *kind == "func");
    if (auto alias = e.getAlias())
      info.alias = alias->str();
    else
      info.alias = e.getSym().str();

    if (!info.isFunc) {
      StringRef dir;
      if (auto d = e.getDirection())
        dir = *d;
      // "in" = host -> device = writable from host.
      // "out" = device -> host = readable only.
      info.writable = (dir != "out");
      Type eltTy;
      auto it = varElt.find(e.getSym());
      if (it != varElt.end())
        eltTy = it->second;
      info.eltName = eltTy ? cslEltName(eltTy).str() : std::string("f32");
    }
    out.push_back(std::move(info));
  });
  return out;
}

/// Build the layout.csl wrapper for a width x height grid that wires
/// memcpy_params into every tile running `<progName>.csl`, and declares
/// every host-visible symbol via `@export_name` so cslc --memcpy can wire
/// them up for the host memcpy subsystem.
static std::string makeLayoutCsl(xilinx::csl::WaferOp wafer, int64_t width,
                                 int64_t height,
                                 const std::string &progName) {
  std::string out;
  llvm::raw_string_ostream os(out);
  os << "// Generated by air-translate --emit-csl\n";
  os << "// CSL layout wrapper that wires `<memcpy/get_params>` into each PE\n";
  os << "// and declares host-visible symbols via `@export_name`.\n";
  os << "// Compile with:\n";
  os << "//   cslc layout.csl --arch=wse3 --fabric-dims=8,3 --fabric-offsets="
        "4,1 \\\n";
  os << "//        -o compiled --memcpy --channels 1\n\n";
  os << "const memcpy = @import_module(\"<memcpy/get_params>\", .{\n";
  os << "  .width = " << width << ",\n";
  os << "  .height = " << height << ",\n";
  os << "});\n\n";
  os << "layout {\n";
  os << "  @set_rectangle(" << width << ", " << height << ");\n";
  for (int64_t y = 0; y < height; ++y)
    for (int64_t x = 0; x < width; ++x)
      os << "  @set_tile_code(" << x << ", " << y << ", \"" << progName
         << ".csl\", .{ .memcpy_params = memcpy.get_params(" << x << ") });\n";
  os << "\n";

  // Declare host-visible exports.
  auto exports = collectExports(wafer);
  for (const auto &e : exports) {
    if (e.isFunc) {
      os << "  @export_name(\"" << e.alias << "\", fn()void);\n";
    } else {
      os << "  @export_name(\"" << e.alias << "\", [*]" << e.eltName << ", "
         << (e.writable ? "true" : "false") << ");\n";
    }
  }
  os << "}\n";
  return out;
}

/// Determine (width, height) for the layout from csl.layout, falling back to
/// 1x1 if absent. Also grab the program name.
static void probeLayout(xilinx::csl::WaferOp wafer, std::string &progName,
                        int64_t &width, int64_t &height) {
  progName = "program";
  width = 1;
  height = 1;
  wafer.walk([&](xilinx::csl::ProgramOp p) {
    if (progName == "program") progName = p.getSymName().str();
  });
  wafer.walk([&](xilinx::csl::LayoutOp layout) {
    int64_t w = 0, h = 0;
    if (auto attr = layout->getAttrOfType<IntegerAttr>("width"))
      w = attr.getInt();
    if (auto attr = layout->getAttrOfType<IntegerAttr>("height"))
      h = attr.getInt();
    if (w > 0) width = w;
    if (h > 0) height = h;
  });
}

/// Emit all five files for a single wafer into `<parentDir>/<waferName>/`.
static LogicalResult emitOneWafer(ModuleOp module,
                                  xilinx::csl::WaferOp wafer,
                                  llvm::StringRef parentDir,
                                  llvm::raw_ostream &statusOs) {
  std::string waferName = wafer.getSymName().str();

  // Create <parentDir>/<waferName>/
  llvm::SmallString<128> waferDir(parentDir);
  llvm::sys::path::append(waferDir, waferName);
  std::error_code ec = llvm::sys::fs::create_directories(
      llvm::StringRef(waferDir.data(), waferDir.size()));
  if (ec) {
    module.emitError() << "cannot create wafer dir '" << waferDir.c_str()
                       << "': " << ec.message();
    return failure();
  }

  std::string progName;
  int64_t layoutW, layoutH;
  probeLayout(wafer, progName, layoutW, layoutH);

  // Helper: open a file inside the wafer subdirectory.
  auto openOut = [&](const std::string &filename,
                     std::unique_ptr<llvm::raw_fd_ostream> &outPtr)
      -> LogicalResult {
    llvm::SmallString<128> path(waferDir);
    llvm::sys::path::append(path, filename);
    std::error_code fec;
    outPtr = std::make_unique<llvm::raw_fd_ostream>(
        llvm::StringRef(path.data(), path.size()), fec,
        llvm::sys::fs::OF_Text);
    if (fec) {
      module.emitError() << "cannot open '" << path.c_str()
                         << "': " << fec.message();
      return failure();
    }
    return success();
  };

  // Emit the three Python/CSL body files.
  std::unique_ptr<llvm::raw_fd_ostream> progOs, layoutPyOs, hostOs;
  if (failed(openOut(progName + ".csl", progOs))) return failure();
  if (failed(openOut("csl_layout.py", layoutPyOs))) return failure();
  if (failed(openOut("run.py", hostOs))) return failure();

  if (failed(runProgramEmitter(wafer, *progOs))) return failure();
  if (failed(runLayoutEmitter(wafer, *layoutPyOs))) return failure();
  if (failed(runHostEmitter(wafer, *hostOs))) return failure();
  progOs.reset();
  layoutPyOs.reset();
  hostOs.reset();

  // Emit layout.csl wrapper.
  {
    llvm::SmallString<128> path(waferDir);
    llvm::sys::path::append(path, "layout.csl");
    std::error_code fec;
    llvm::raw_fd_ostream out(llvm::StringRef(path.data(), path.size()), fec,
                             llvm::sys::fs::OF_Text);
    if (fec) {
      module.emitError() << "cannot open '" << path.c_str()
                         << "': " << fec.message();
      return failure();
    }
    out << makeLayoutCsl(wafer, layoutW, layoutH, progName);
  }

  // Emit commands_wse3.sh.
  int64_t fabricX = layoutW + 7;
  int64_t fabricY = layoutH + 2;
  std::string cmdWse3;
  {
    llvm::raw_string_ostream ss(cmdWse3);
    ss << "#!/usr/bin/env bash\n";
    ss << "# Generated by air-translate --emit-csl\n";
    ss << "# One-command runner: cslc compile + cs_python host driver.\n";
    ss << "set -e\n";
    ss << "DIR=\"$(cd \"$(dirname \"${BASH_SOURCE[0]}\")\" && pwd)\"\n";
    ss << "cd \"$DIR\"\n";
    ss << "cslc layout.csl --arch=wse3 \\\n";
    ss << "     --fabric-dims=" << fabricX << "," << fabricY << " \\\n";
    ss << "     --fabric-offsets=4,1 \\\n";
    ss << "     -o compiled --memcpy --channels 1\n";
    ss << "cs_python run.py --name=compiled\n";
  }
  if (failed(writeExecutable(module, llvm::StringRef(waferDir.data(),
                                                     waferDir.size()),
                             "commands_wse3.sh", cmdWse3)))
    return failure();

  statusOs << "  " << llvm::StringRef(waferDir.data(), waferDir.size())
           << "/{" << progName
           << ".csl, layout.csl, csl_layout.py, run.py, commands_wse3.sh}\n";
  return success();
}

static LogicalResult emitAll(ModuleOp module, llvm::raw_ostream &os) {
  // `os` is only used for a short status block — the real outputs go to
  // files under --output-dir/<wafer-name>/.
  if (EmitCslOutputDir.empty()) {
    module.emitError() << "--emit-csl requires --output-dir=<path>";
    return failure();
  }

  std::error_code ec =
      llvm::sys::fs::create_directories(EmitCslOutputDir.getValue());
  if (ec) {
    module.emitError() << "cannot create output-dir '" << EmitCslOutputDir
                       << "': " << ec.message();
    return failure();
  }

  // Collect all wafers in source order.
  llvm::SmallVector<xilinx::csl::WaferOp, 4> wafers;
  module.walk([&](xilinx::csl::WaferOp w) { wafers.push_back(w); });
  if (wafers.empty()) {
    module.emitError() << "--emit-csl: no csl.wafer found in module";
    return failure();
  }

  os << "emitted:\n";
  for (auto wafer : wafers) {
    if (failed(emitOneWafer(module, wafer, EmitCslOutputDir.getValue(), os)))
      return failure();
  }
  return success();
}

} // namespace

void registerCSLEmitAllTranslation() {
  static TranslateFromMLIRRegistration reg(
      "emit-csl",
      "Emit all CSL files (program, layout, host, commands_wse3.sh) "
      "into --output-dir. Writes one subdirectory per csl.wafer.",
      emitAll,
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

/// Register all four CSL translations (program, layout, host, emit-all).
void registerCSLEmitTranslations() {
  registerCSLProgramTranslation();
  registerCSLLayoutTranslation();
  registerCSLHostTranslation();
  registerCSLEmitAllTranslation();
}

} // namespace csl
} // namespace xilinx
