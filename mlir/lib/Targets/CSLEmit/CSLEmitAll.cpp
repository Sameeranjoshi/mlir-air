//===- CSLEmitAll.cpp - Unified --emit-csl command -------------*- C++ -*-===//
//
// Part of the air-to-csl project.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
//
// Registers --emit-csl, which writes all three CSL output files
// (<program>.csl, csl_layout.py, run.py) into --output-dir=<path>.
//
// Also provides registerCSLEmitTranslations() that registers all four
// CSL translations in one call.
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
LogicalResult runProgramEmitter(ModuleOp module, llvm::raw_ostream &os);
LogicalResult runLayoutEmitter(ModuleOp module, llvm::raw_ostream &os);
LogicalResult runHostEmitter(ModuleOp module, llvm::raw_ostream &os);

// Forward-declare the individual registration entry points.
void registerCSLProgramTranslation();
void registerCSLLayoutTranslation();
void registerCSLHostTranslation();

namespace {

static llvm::cl::opt<std::string> EmitCslOutputDir(
    "output-dir",
    llvm::cl::desc("Output directory for --emit-csl (required)"),
    llvm::cl::init(""));

static LogicalResult emitAll(ModuleOp module, llvm::raw_ostream &os) {
  // `os` is only used for a short status line — the three real outputs go
  // to files under --output-dir.
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

  // Determine program file name from the first csl.program's sym_name.
  std::string progName;
  module.walk([&](xilinx::csl::ProgramOp p) {
    if (progName.empty()) progName = p.getSymName().str();
  });
  if (progName.empty())
    progName = "program";

  auto openOut = [&](const std::string &filename,
                     std::unique_ptr<llvm::raw_fd_ostream> &outPtr)
      -> LogicalResult {
    llvm::SmallString<128> path(EmitCslOutputDir.getValue());
    llvm::sys::path::append(path, filename);
    std::error_code ec;
    outPtr = std::make_unique<llvm::raw_fd_ostream>(
        llvm::StringRef(path.data(), path.size()), ec, llvm::sys::fs::OF_Text);
    if (ec) {
      module.emitError() << "cannot open '" << path.c_str()
                         << "': " << ec.message();
      return failure();
    }
    return success();
  };

  std::unique_ptr<llvm::raw_fd_ostream> progOs, layoutOs, hostOs;
  if (failed(openOut(progName + ".csl", progOs))) return failure();
  if (failed(openOut("csl_layout.py", layoutOs))) return failure();
  if (failed(openOut("run.py", hostOs))) return failure();

  if (failed(runProgramEmitter(module, *progOs))) return failure();
  if (failed(runLayoutEmitter(module, *layoutOs))) return failure();
  if (failed(runHostEmitter(module, *hostOs))) return failure();

  os << "emitted: " << EmitCslOutputDir << "/" << progName << ".csl, "
     << EmitCslOutputDir << "/csl_layout.py, " << EmitCslOutputDir
     << "/run.py\n";
  return success();
}

} // namespace

void registerCSLEmitAllTranslation() {
  static TranslateFromMLIRRegistration reg(
      "emit-csl",
      "Emit all three CSL files (program, layout, host) into --output-dir",
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
