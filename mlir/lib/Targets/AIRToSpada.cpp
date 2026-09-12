//===- AIRToSpada.cpp -------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
//
// air-translate --air-to-spada : AIR (after -air-place-herds-by-token) ->
// SpaDA Spatial IR text (.sptl) for the Cerebras WSE.
//
// See docs/SPADA_EMITTER_SPEC.md for the complete specification this file
// implements. Section numbers referenced in comments below are sections of
// that spec.
//
//===----------------------------------------------------------------------===//

#include "air/Dialect/AIR/AIRDialect.h"
#include "air/Util/Util.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/Tools/mlir-translate/Translation.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_ostream.h"

#include <map>
#include <optional>
#include <set>
#include <string>
#include <vector>

using namespace mlir;
using namespace xilinx;

#define DEBUG_TYPE "air-to-spada"

namespace {

//===----------------------------------------------------------------------===//
// Index evaluation under a per-tile environment.
//
// Copied from AIRPlaceHerdsByToken.cpp (file-static there; duplicated here
// per the spec rather than exported), with executesOnTile / getPartition /
// getPartitionOfHerdArg adapted for this file's needs.
//===----------------------------------------------------------------------===//

using Env = DenseMap<Value, int64_t>;

static std::optional<int64_t> evalIndex(Value v, const Env &env);

static std::optional<SmallVector<int64_t>>
evalAffineMap(AffineMap map, ValueRange operands, const Env &env) {
  SmallVector<Attribute> consts;
  Builder b(map.getContext());
  for (Value o : operands) {
    auto c = evalIndex(o, env);
    if (!c)
      return std::nullopt;
    consts.push_back(b.getIndexAttr(*c));
  }
  SmallVector<Attribute> results;
  if (failed(map.constantFold(consts, results)))
    return std::nullopt;
  SmallVector<int64_t> out;
  for (Attribute r : results)
    out.push_back(cast<IntegerAttr>(r).getInt());
  return out;
}

static std::optional<int64_t> evalIndex(Value v, const Env &env) {
  auto it = env.find(v);
  if (it != env.end())
    return it->second;
  if (auto c = getConstantIntValue(v))
    return c;
  Operation *def = v.getDefiningOp();
  if (!def)
    return std::nullopt;
  if (auto apply = dyn_cast<affine::AffineApplyOp>(def)) {
    auto r = evalAffineMap(apply.getAffineMap(), apply.getMapOperands(), env);
    if (!r)
      return std::nullopt;
    return (*r)[0];
  }
  if (auto cast = dyn_cast<arith::IndexCastOp>(def))
    return evalIndex(cast.getIn(), env);
  auto binary = [&](auto op, auto fn) -> std::optional<int64_t> {
    auto l = evalIndex(op.getLhs(), env);
    auto r = evalIndex(op.getRhs(), env);
    if (!l || !r)
      return std::nullopt;
    return fn(*l, *r);
  };
  if (auto op = dyn_cast<arith::AddIOp>(def))
    return binary(op, [](int64_t a, int64_t b) { return a + b; });
  if (auto op = dyn_cast<arith::SubIOp>(def))
    return binary(op, [](int64_t a, int64_t b) { return a - b; });
  if (auto op = dyn_cast<arith::MulIOp>(def))
    return binary(op, [](int64_t a, int64_t b) { return a * b; });
  return std::nullopt;
}

// Does the integer set of `ifOp` hold under `env`? nullopt if not evaluable.
static std::optional<bool> evalAffineIf(affine::AffineIfOp ifOp,
                                         const Env &env) {
  IntegerSet set = ifOp.getIntegerSet();
  for (unsigned i = 0, e = set.getNumConstraints(); i < e; ++i) {
    AffineMap m = AffineMap::get(set.getNumDims(), set.getNumSymbols(),
                                  set.getConstraint(i));
    auto r = evalAffineMap(m, ifOp.getOperands(), env);
    if (!r)
      return std::nullopt;
    int64_t val = (*r)[0];
    if (set.isEq(i) ? (val != 0) : (val < 0))
      return false;
  }
  return true;
}

static air::PartitionAttr getPartition(Value memref) {
  auto alloc = memref.getDefiningOp<memref::AllocOp>();
  if (!alloc)
    return nullptr;
  return alloc->getAttrOfType<air::PartitionAttr>(
      air::PartitionAttr::getAttrName());
}

// Resolve a herd/segment kernel argument to the partition attribute of the
// L2 alloc it is tied to (through segment args as well).
[[maybe_unused]] static air::PartitionAttr
getPartitionOfHerdArg(air::HerdOp herd, Value v) {
  auto arg = dyn_cast<BlockArgument>(v);
  if (!arg || arg.getOwner()->getParentOp() != herd.getOperation())
    return nullptr;
  Value operand = herd.getTiedKernelOperand(arg);
  if (!operand)
    return nullptr;
  if (auto p = getPartition(operand))
    return p;
  if (auto segArg = dyn_cast<BlockArgument>(operand))
    if (auto seg = dyn_cast<air::SegmentOp>(segArg.getOwner()->getParentOp()))
      if (Value segOperand = seg.getTiedKernelOperand(segArg))
        return getPartition(segOperand);
  return nullptr;
}

// Like getPartitionOfHerdArg, but returns the L2 alloc Value itself (so the
// caller can look it up in a name/index table), not just its attribute.
static Value getL2AllocOfHerdArg(air::HerdOp herd, Value v) {
  auto arg = dyn_cast<BlockArgument>(v);
  if (!arg || arg.getOwner()->getParentOp() != herd.getOperation())
    return nullptr;
  Value operand = herd.getTiedKernelOperand(arg);
  if (!operand)
    return nullptr;
  if (getPartition(operand))
    return operand;
  if (auto segArg = dyn_cast<BlockArgument>(operand))
    if (auto seg = dyn_cast<air::SegmentOp>(segArg.getOwner()->getParentOp()))
      if (Value segOperand = seg.getTiedKernelOperand(segArg))
        if (getPartition(segOperand))
          return segOperand;
  return nullptr;
}

//===----------------------------------------------------------------------===//
// Small formatting helpers.
//===----------------------------------------------------------------------===//

// Prints a range [lo:hi) with stride `stride` using SpaDA's shorthand rules:
// a single value prints as `a`, a unit stride prints as `a:b`, anything else
// prints as `a:b:s`.
static std::string fmtRange(int64_t lo, int64_t hi, int64_t stride = 1) {
  if (hi - lo == 1)
    return std::to_string(lo);
  if (stride == 1)
    return (Twine(lo) + ":" + Twine(hi)).str();
  return (Twine(lo) + ":" + Twine(hi) + ":" + Twine(stride)).str();
}

static std::string fmtRect(int64_t x0, int64_t x1, int64_t y0, int64_t y1,
                            int64_t sx = 1, int64_t sy = 1) {
  return (Twine(fmtRange(x0, x1, sx)) + ", " + Twine(fmtRange(y0, y1, sy)))
      .str();
}

static int64_t mod2(int64_t v) { return ((v % 2) + 2) % 2; }

// Spec v2 §1: formats a ranked SpaDA array declaration shape, e.g. `8, 8`.
static std::string formatDims(ArrayRef<int64_t> shape) {
  std::string s;
  for (size_t i = 0; i < shape.size(); ++i) {
    if (i)
      s += ", ";
    s += std::to_string(shape[i]);
  }
  return s;
}

// Formats a float constant the way SpaDA literals look (`0.0`, `1.5`).
static std::string fmtFloat(const APFloat &v) {
  double d = v.convertToDouble();
  if (d == static_cast<int64_t>(d))
    return (Twine(static_cast<int64_t>(d)) + ".0").str();
  std::string s;
  llvm::raw_string_ostream os(s);
  os << llvm::format("%g", d);
  return s;
}

// Maps a memref element type to a SpaDA scalar type name.
static FailureOr<std::string> spadaScalarType(Operation *errOp, Type elemTy) {
  if (elemTy.isF32())
    return std::string("f32");
  if (elemTy.isF16())
    return std::string("f16");
  if (auto it = dyn_cast<IntegerType>(elemTy))
    if (it.getWidth() == 32)
      return std::string("i32");
  return errOp->emitError("unsupported element type for a SpaDA stream/array: ")
         << elemTy;
}

//===----------------------------------------------------------------------===//
// Structural model built by the analysis pass.
//===----------------------------------------------------------------------===//

struct L2AllocInfo {
  Value alloc;
  air::PartitionAttr part;
  MemRefType ty;
  std::string name; // "l2_<k>"
  int64_t blocksize = 0;
  SmallVector<int64_t> blockShape; // spec v2 §1: declared/indexed rank
  int64_t ox0 = 0, ox1 = 0, oy0 = 0, oy1 = 0; // owner rectangle (global)
};

struct HerdInfo {
  air::HerdOp op;
  int64_t sx = 1, sy = 1;
  int64_t xloc = 0, yloc = 0;
  std::string name;
  // Spec v2 §1: an L1 buffer that is ever the destination of an
  // `air.channel.get` anywhere in the herd is emitted flat; every other L1
  // buffer keeps its memref rank. Defaults to false (ranked) for allocs not
  // present in the map.
  DenseMap<Value, bool> l1IsFlat;
  SmallVector<Value> l1Allocs;           // in program order
  DenseMap<Value, std::string> l1Names;  // alloc result -> "<h>_l1_<j>"
};

enum class SegDmaDir { L3ToL2, L2ToL3 };

struct SegDmaInfo {
  SegDmaDir dir;
  unsigned funcArgIdx;
  unsigned l2Idx;
};

struct KernelArgInfo {
  bool hasIn = false, hasOut = false;
  unsigned l2InIdx = 0, l2OutIdx = 0;
  MemRefType ty;
};

// One item of the top-level program order: either a run of consecutive
// segment-level DMAs (all same direction) or a single herd.
struct ProgramItem {
  bool isHerd;
  unsigned herdIdx = 0;       // valid if isHerd
  SmallVector<unsigned> dmas; // indices into segDmas, valid if !isHerd
};

//===----------------------------------------------------------------------===//
// Channel classification (spec 3.5).
//===----------------------------------------------------------------------===//

struct ChannelInfo {
  enum Kind { Multicast, Ring } kind;
  std::string elemType;
  // Multicast:
  int axis = 1;    // 0 -> dx varies, 1 -> dy varies
  int sign = 1;     // +1 ascending, -1 descending
  int64_t n = 0;    // number of receivers
  int channelNum = 0;
  // Ring / chain along one axis (spec v2 §4; a pure chain with no wrap edge
  // reproduces v1's unit-chain behavior):
  int64_t dx = 0, dy = 0; // base edge net offset (any length)
  int64_t baseStepX = 0, baseStepY = 0; // unit hop direction of the base edge
  int64_t baseHops = 1;                 // |dx| + |dy|
  int evenChannelNum = 0, oddChannelNum = 0;
  bool hasWrap = false;
  int64_t wrapDx = 0, wrapDy = 0;         // long edge's net (Lx, Ly)
  int64_t wrapStepX = 0, wrapStepY = 0;   // unit hop direction for the wrap
  int64_t wrapHops = 0;                   // number of hops (|Lx| + |Ly|)
  int wrapChannelNum = 0;
  // Per-tile stream selection for a Ring channel, keyed by global (x, y).
  // 0 = even, 1 = odd, 2 = wrap.
  std::map<std::pair<int64_t, int64_t>, int> outStream;
  std::map<std::pair<int64_t, int64_t>, int> inStream;
};

static const char *ringStreamSuffix(int sel) {
  return sel == 0 ? "even" : sel == 1 ? "odd" : "wrap";
}

static FailureOr<int> lookupRingStream(
    Operation *errOp, const std::map<std::pair<int64_t, int64_t>, int> &m,
    int64_t gx, int64_t gy, const char *which) {
  auto it = m.find({gx, gy});
  if (it == m.end())
    return errOp->emitOpError("internal error: tile has no assigned ")
           << which << " stream for this ring channel";
  return it->second;
}

using ChannelMap = llvm::StringMap<ChannelInfo>;

// A tile's put/get record for a single channel: global (x, y) tile plus the
// evaluated index vector.
struct SiteRecord {
  int64_t gx, gy;
  SmallVector<int64_t> idx;
};

// Spec v2 §2: a `completion cN = send/receive(...)` produced by an async
// channel put/get, recorded so a later (no-result) `air.wait_all` can be
// lowered to `await cN`. `block` is the SpaDA-emitting Block the completion
// was declared in, so an `air.wait_all` outside that block/loop is an error
// (SpaDA requires the await to be in the same body as the completion).
struct CompletionRecord {
  std::string name;
  int depth; // affine.for nesting depth (SpaDA scope) it was declared in.
};

//===----------------------------------------------------------------------===//
// The emitter.
//===----------------------------------------------------------------------===//

class SpadaEmitter {
public:
  SpadaEmitter(ModuleOp module, raw_ostream &os) : module(module), os(os) {}

  LogicalResult run();

private:
  ModuleOp module;
  raw_ostream &os;

  func::FuncOp funcOp;
  air::LaunchOp launchOp;
  air::SegmentOp segmentOp;

  SmallVector<L2AllocInfo> l2Allocs;
  DenseMap<Value, unsigned> l2Index; // alloc Value -> index into l2Allocs

  SmallVector<HerdInfo> herds;
  DenseMap<Operation *, unsigned> herdIndex;

  SmallVector<SegDmaInfo> segDmas;
  SmallVector<ProgramItem> program;

  llvm::MapVector<unsigned, KernelArgInfo> kernelArgs; // func arg idx -> info

  int nextChannel = 0;

  // ---- Analysis (pass 1) ----
  LogicalResult collectStructure();
  LogicalResult collectL2Allocs();
  LogicalResult collectHerds();
  LogicalResult collectSegDmasAndProgram();
  LogicalResult resolveKernelArgs();

  // ---- Emission (pass 2) ----
  LogicalResult emitAll();
  void emitPlaceBlock(int64_t ux0, int64_t ux1, int64_t uy0, int64_t uy1);
  LogicalResult emitSegDmaGroup(ArrayRef<unsigned> dmaIdxs);
  LogicalResult emitHerdPhase(HerdInfo &herd);

  // ---- Channel classification ----
  FailureOr<ChannelMap> classifyHerdChannels(HerdInfo &herd,
                                              SmallVectorImpl<std::string> &order);
  LogicalResult collectPutGet(
      Block &block, Env env, HerdInfo &herd, int64_t gx, int64_t gy,
      llvm::StringMap<SmallVector<SiteRecord>> &puts,
      llvm::StringMap<SmallVector<SiteRecord>> &gets,
      SmallVectorImpl<std::string> &order,
      llvm::StringMap<air::ChannelOp> &channelOpOf,
      llvm::StringMap<Type> &elemTypeOf);

  // ---- Role grouping ----
  LogicalResult buildRoleKey(Block &block, Env &env, HerdInfo &herd,
                              int64_t lx, int64_t ly, ChannelMap &channels,
                              std::string &key);

  // ---- Code generation for a herd body ----
  LogicalResult emitHerdBody(Block &block, HerdInfo &herd, Env &env,
                              int64_t lx, int64_t ly, ChannelMap &channels,
                              DenseMap<Value, std::string> &loopVarNames,
                              int depth, raw_ostream &o,
                              DenseMap<Value, CompletionRecord> &completions,
                              int &nextCompletionId);
  FailureOr<std::string> printExpr(Value v, HerdInfo &herd, int64_t lx,
                                    int64_t ly,
                                    DenseMap<Value, std::string> &loopVarNames);
  FailureOr<std::string>
  printAffineExpr(AffineExpr e, ValueRange operands, HerdInfo &herd,
                   int64_t lx, int64_t ly,
                   DenseMap<Value, std::string> &loopVarNames);
  FailureOr<std::string> flattenAccess(Operation *op, Value memref,
                                        ArrayRef<Value> plainIndices,
                                        std::optional<AffineMap> map,
                                        ValueRange mapOperands, HerdInfo &herd,
                                        int64_t lx, int64_t ly,
                                        DenseMap<Value, std::string> &loopVarNames,
                                        std::string &bufName);

  // Resolve a Value used inside a herd body to a "buffer name": either an L1
  // alloc local to the herd, or a partitioned L2 alloc (herd argument).
  std::string lookupL1Name(HerdInfo &herd, Value memref) {
    auto it = herd.l1Names.find(memref);
    return it == herd.l1Names.end() ? std::string() : it->second;
  }
};

//===----------------------------------------------------------------------===//
// Pass 1: structural analysis.
//===----------------------------------------------------------------------===//

LogicalResult SpadaEmitter::collectStructure() {
  SmallVector<func::FuncOp> funcs(module.getOps<func::FuncOp>());
  if (funcs.size() != 1)
    return module.emitError("air-to-spada requires exactly one func.func in "
                             "the module, found ")
           << funcs.size();
  funcOp = funcs.front();

  SmallVector<air::LaunchOp> launches;
  funcOp.walk([&](air::LaunchOp l) { launches.push_back(l); });
  if (launches.size() != 1)
    return funcOp.emitError("expected exactly one air.launch, found ")
           << launches.size();
  launchOp = launches.front();

  for (Value sz : launchOp.getSizeOperands()) {
    auto c = getConstantIntValue(sz);
    if (!c || *c != 1)
      return launchOp.emitOpError(
          "expected a 1x1 (or absent) launch iteration space");
  }

  SmallVector<air::SegmentOp> segments;
  launchOp.walk([&](air::SegmentOp s) { segments.push_back(s); });
  if (segments.size() != 1)
    return launchOp.emitOpError("expected exactly one air.segment, found ")
           << segments.size();
  segmentOp = segments.front();

  if (failed(collectL2Allocs()))
    return failure();
  if (failed(collectHerds()))
    return failure();
  if (failed(collectSegDmasAndProgram()))
    return failure();
  if (failed(resolveKernelArgs()))
    return failure();
  return success();
}

LogicalResult SpadaEmitter::collectL2Allocs() {
  for (Operation &op : segmentOp.getBody().front()) {
    auto alloc = dyn_cast<memref::AllocOp>(op);
    if (!alloc)
      continue;
    auto ty = cast<MemRefType>(alloc.getType());
    auto ms = air::getMemorySpace(cast<BaseMemRefType>(ty));
    if (!ms || *ms != air::MemorySpace::L2)
      continue; // not an L2 alloc; ignore (e.g. L1 escapees are not legal
                // here but that is caught elsewhere if ever touched).
    auto part = alloc->getAttrOfType<air::PartitionAttr>(
        air::PartitionAttr::getAttrName());
    if (!part)
      return alloc.emitOpError(
          "L2 (memory space 1) memref.alloc without #air.partition: no "
          "shared memory on this target");

    L2AllocInfo info;
    info.alloc = alloc.getResult();
    info.part = part;
    info.ty = ty;

    ArrayRef<int64_t> shape = ty.getShape();
    ArrayRef<int64_t> block = part.getBlock();
    if (block.size() != shape.size())
      return alloc.emitOpError("air.partition block rank does not match the "
                                "memref rank");
    SmallVector<int64_t> gridDims(shape.size());
    int64_t blocksize = 1;
    for (size_t d = 0; d < shape.size(); ++d) {
      if (block[d] <= 0 || shape[d] % block[d] != 0)
        return alloc.emitOpError("memref shape is not a multiple of the "
                                  "partition block size");
      gridDims[d] = shape[d] / block[d];
      blocksize *= block[d];
    }
    info.blocksize = blocksize;
    info.blockShape.assign(block.begin(), block.end());

    // Enumerate every block coordinate and evaluate its owner.
    int64_t nBlocks = 1;
    for (int64_t g : gridDims)
      nBlocks *= g;
    std::set<std::pair<int64_t, int64_t>> owners;
    int64_t minx = INT64_MAX, maxx = INT64_MIN, miny = INT64_MAX,
            maxy = INT64_MIN;
    SmallVector<int64_t> coord(gridDims.size(), 0);
    for (int64_t i = 0; i < nBlocks; ++i) {
      auto owner = part.getOwnerOfBlock(coord);
      if (!owner || owner->size() != 2)
        return alloc.emitOpError(
            "air.partition owner map must fold to a 2D tile coordinate");
      owners.insert({(*owner)[0], (*owner)[1]});
      minx = std::min(minx, (*owner)[0]);
      maxx = std::max(maxx, (*owner)[0]);
      miny = std::min(miny, (*owner)[1]);
      maxy = std::max(maxy, (*owner)[1]);
      for (int64_t d = gridDims.size() - 1; d >= 0; --d) {
        if (++coord[d] < gridDims[d])
          break;
        coord[d] = 0;
      }
    }
    int64_t rectArea = (maxx - minx + 1) * (maxy - miny + 1);
    if ((int64_t)owners.size() != rectArea)
      return alloc.emitOpError(
          "owner tiles of #air.partition do not form a rectangle");
    info.ox0 = minx;
    info.ox1 = maxx + 1;
    info.oy0 = miny;
    info.oy1 = maxy + 1;
    info.name = ("l2_" + Twine(l2Allocs.size())).str();

    l2Index[info.alloc] = l2Allocs.size();
    l2Allocs.push_back(info);
  }
  return success();
}

LogicalResult SpadaEmitter::collectHerds() {
  for (auto herd : segmentOp.getOps<air::HerdOp>()) {
    HerdInfo info;
    info.op = herd;
    SmallVector<int64_t, 2> sizes;
    for (Value s : herd.getSizeOperands()) {
      auto c = getConstantIntValue(s);
      if (!c)
        return herd.emitOpError("herd size must be a compile-time constant");
      sizes.push_back(*c);
    }
    if (sizes.empty() || sizes.size() > 2)
      return herd.emitOpError("expected a 1D or 2D herd");
    info.sx = sizes[0];
    info.sy = sizes.size() > 1 ? sizes[1] : 1;
    auto xloc = herd.getColOffset();
    auto yloc = herd.getRowOffset();
    if (!xloc || !yloc)
      return herd.emitOpError(
          "missing x_loc/y_loc; run -air-place-herds-by-token first");
    info.xloc = *xloc;
    info.yloc = *yloc;
    auto symName = herd.getSymName();
    info.name = (!symName || symName->empty())
                    ? ("herd" + Twine(herds.size())).str()
                    : symName->str();

    // L1 allocs, in program order.
    herd.walk([&](memref::AllocOp alloc) {
      auto ty = cast<MemRefType>(alloc.getType());
      auto ms = air::getMemorySpace(cast<BaseMemRefType>(ty));
      if (ms && *ms == air::MemorySpace::L1) {
        std::string name =
            (Twine(info.name) + "_l1_" + Twine(info.l1Allocs.size())).str();
        info.l1Names[alloc.getResult()] = name;
        info.l1Allocs.push_back(alloc.getResult());
      }
    });

    // Spec v2 §1: mark every L1 buffer that is ever a channel.get
    // destination in this herd as flat.
    herd.walk([&](air::ChannelGetOp get) { info.l1IsFlat[get.getDst()] = true; });

    herdIndex[herd.getOperation()] = herds.size();
    herds.push_back(info);
  }
  return success();
}

LogicalResult SpadaEmitter::collectSegDmasAndProgram() {
  for (Operation &op : segmentOp.getBody().front()) {
    if (auto herd = dyn_cast<air::HerdOp>(op)) {
      ProgramItem item;
      item.isHerd = true;
      item.herdIdx = herdIndex.lookup(herd.getOperation());
      program.push_back(item);
      continue;
    }
    auto dma = dyn_cast<air::DmaMemcpyNdOp>(op);
    if (!dma)
      continue;
    Value src = dma.getSrcMemref(), dst = dma.getDstMemref();
    // One side is a segment kernel argument (L3), the other a partitioned
    // L2 alloc we already collected.
    unsigned srcL2 = -1u, dstL2 = -1u;
    if (auto it = l2Index.find(src); it != l2Index.end())
      srcL2 = it->second;
    if (auto it = l2Index.find(dst); it != l2Index.end())
      dstL2 = it->second;
    if (srcL2 == -1u && dstL2 == -1u)
      continue; // not a segment-level scatter/gather we care about.
    if (srcL2 != -1u && dstL2 != -1u)
      return dma.emitOpError(
          "segment-level dma_memcpy_nd between two partitioned L2 buffers "
          "is not supported");

    auto checkWholeBuffer = [&](ArrayRef<OpFoldResult> offs) {
      return offs.empty();
    };
    if (!checkWholeBuffer(dma.getMixedSrcOffsets()) ||
        !checkWholeBuffer(dma.getMixedSrcSizes()) ||
        !checkWholeBuffer(dma.getMixedSrcStrides()) ||
        !checkWholeBuffer(dma.getMixedDstOffsets()) ||
        !checkWholeBuffer(dma.getMixedDstSizes()) ||
        !checkWholeBuffer(dma.getMixedDstStrides()))
      return dma.emitOpError("segment-level dma_memcpy_nd to/from a "
                              "partitioned L2 buffer must be whole-buffer "
                              "([] [] [] on both sides)");

    Value l3Side = dstL2 != -1u ? src : dst;
    auto l3Arg = dyn_cast<BlockArgument>(l3Side);
    if (!l3Arg || l3Arg.getOwner()->getParentOp() != segmentOp.getOperation())
      return dma.emitOpError(
          "expected the non-L2 side of a segment-level dma_memcpy_nd to be "
          "a segment kernel argument");
    Value segOperand = segmentOp.getTiedKernelOperand(l3Arg);
    if (!segOperand)
      return dma.emitOpError("could not trace the segment argument to a "
                              "launch operand");
    auto launchArg = dyn_cast<BlockArgument>(segOperand);
    if (!launchArg ||
        launchArg.getOwner()->getParentOp() != launchOp.getOperation())
      return dma.emitOpError(
          "expected the segment argument to be tied to a launch argument");
    Value launchOperand = launchOp.getTiedKernelOperand(launchArg);
    if (!launchOperand)
      return dma.emitOpError("could not trace the launch argument to a "
                              "func.func argument");
    auto funcArg = dyn_cast<BlockArgument>(launchOperand);
    if (!funcArg || funcArg.getOwner() != &funcOp.getBody().front())
      return dma.emitOpError(
          "expected the launch argument to be a func.func argument");
    auto srcTy = cast<BaseMemRefType>(l3Side.getType());
    auto ms = air::getMemorySpace(srcTy);
    if (ms && *ms != air::MemorySpace::L3)
      return dma.emitOpError("expected the L3 side to be in memory space 0");

    SegDmaInfo info;
    info.dir = dstL2 != -1u ? SegDmaDir::L3ToL2 : SegDmaDir::L2ToL3;
    info.funcArgIdx = funcArg.getArgNumber();
    info.l2Idx = dstL2 != -1u ? dstL2 : srcL2;
    unsigned dmaIdx = segDmas.size();
    segDmas.push_back(info);

    // Spec v2 addendum: each segment-level DMA gets its own phase (SpaDA
    // allows at most one compute block per PE per phase; grouping
    // consecutive same-direction DMAs, as v1 did, can put two overlapping
    // compute-block rectangles in one phase).
    ProgramItem item;
    item.isHerd = false;
    item.dmas.push_back(dmaIdx);
    program.push_back(item);
  }
  return success();
}

LogicalResult SpadaEmitter::resolveKernelArgs() {
  for (auto &dma : segDmas) {
    auto &info = kernelArgs[dma.funcArgIdx];
    info.ty = cast<MemRefType>(funcOp.getArgument(dma.funcArgIdx).getType());
    if (dma.dir == SegDmaDir::L3ToL2) {
      info.hasIn = true;
      info.l2InIdx = dma.l2Idx;
    } else {
      info.hasOut = true;
      info.l2OutIdx = dma.l2Idx;
    }
  }
  return success();
}

//===----------------------------------------------------------------------===//
// Channel classification.
//===----------------------------------------------------------------------===//

LogicalResult SpadaEmitter::collectPutGet(
    Block &block, Env env, HerdInfo &herd, int64_t gx, int64_t gy,
    llvm::StringMap<SmallVector<SiteRecord>> &puts,
    llvm::StringMap<SmallVector<SiteRecord>> &gets,
    SmallVectorImpl<std::string> &order,
    llvm::StringMap<air::ChannelOp> &channelOpOf,
    llvm::StringMap<Type> &elemTypeOf) {
  for (Operation &op : block) {
    if (auto ifOp = dyn_cast<affine::AffineIfOp>(op)) {
      auto holds = evalAffineIf(ifOp, env);
      if (!holds)
        return ifOp.emitOpError(
            "condition does not fold to a constant under the tile "
            "environment");
      if (*holds) {
        if (failed(collectPutGet(*ifOp.getThenBlock(), env, herd, gx, gy,
                                  puts, gets, order, channelOpOf, elemTypeOf)))
          return failure();
      } else if (ifOp.hasElse()) {
        if (failed(collectPutGet(*ifOp.getElseBlock(), env, herd, gx, gy,
                                  puts, gets, order, channelOpOf, elemTypeOf)))
          return failure();
      }
      continue;
    }
    if (auto forOp = dyn_cast<affine::AffineForOp>(op)) {
      if (failed(collectPutGet(forOp.getBody()->getParent()->front(), env,
                                herd, gx, gy, puts, gets, order, channelOpOf,
                                elemTypeOf)))
        return failure();
      continue;
    }
    if (auto put = dyn_cast<air::ChannelPutOp>(op)) {
      std::string name = put.getChanName().str();
      if (!channelOpOf.count(name)) {
        order.push_back(name);
        channelOpOf[name] = air::getChannelDeclarationThroughSymbol(
            cast<air::ChannelInterface>(put.getOperation()));
        elemTypeOf[name] =
            cast<MemRefType>(put.getSrc().getType()).getElementType();
      }
      SiteRecord rec;
      rec.gx = gx;
      rec.gy = gy;
      for (Value idx : put.getIndices()) {
        auto v = evalIndex(idx, env);
        if (!v)
          return put.emitOpError("channel index does not fold to a constant "
                                  "under the tile environment");
        rec.idx.push_back(*v);
      }
      puts[name].push_back(rec);
      continue;
    }
    if (auto get = dyn_cast<air::ChannelGetOp>(op)) {
      std::string name = get.getChanName().str();
      if (!channelOpOf.count(name)) {
        order.push_back(name);
        channelOpOf[name] = air::getChannelDeclarationThroughSymbol(
            cast<air::ChannelInterface>(get.getOperation()));
        elemTypeOf[name] =
            cast<MemRefType>(get.getDst().getType()).getElementType();
      }
      SiteRecord rec;
      rec.gx = gx;
      rec.gy = gy;
      for (Value idx : get.getIndices()) {
        auto v = evalIndex(idx, env);
        if (!v)
          return get.emitOpError("channel index does not fold to a constant "
                                  "under the tile environment");
        rec.idx.push_back(*v);
      }
      gets[name].push_back(rec);
      continue;
    }
    // Everything else (memref ops, dma, arith, ...) is irrelevant to
    // channel enumeration.
  }
  return success();
}

FailureOr<ChannelMap>
SpadaEmitter::classifyHerdChannels(HerdInfo &herd,
                                    SmallVectorImpl<std::string> &order) {
  llvm::StringMap<SmallVector<SiteRecord>> puts, gets;
  llvm::StringMap<air::ChannelOp> channelOpOf;
  llvm::StringMap<Type> elemTypeOf;

  for (int64_t x = 0; x < herd.sx; ++x) {
    for (int64_t y = 0; y < herd.sy; ++y) {
      Env env;
      auto ids = herd.op.getIds();
      auto sizes = herd.op.getSize();
      env[ids[0]] = x;
      env[sizes[0]] = herd.sx;
      if (ids.size() > 1) {
        env[ids[1]] = y;
        env[sizes[1]] = herd.sy;
      }
      if (failed(collectPutGet(herd.op.getBody().front(), env, herd,
                                herd.xloc + x, herd.yloc + y, puts, gets,
                                order, channelOpOf, elemTypeOf)))
        return failure();
    }
  }

  ChannelMap result;
  for (auto &name : order) {
    air::ChannelOp chanOp = channelOpOf[name];
    SmallVector<int64_t> channelSize;
    for (Attribute a : chanOp.getSize())
      channelSize.push_back(cast<IntegerAttr>(a).getInt());
    if (channelSize.empty())
      channelSize.push_back(1);

    auto matches = [&](ArrayRef<int64_t> p, ArrayRef<int64_t> g) {
      if (p.size() != g.size() || p.size() != channelSize.size())
        return false;
      for (size_t d = 0; d < p.size(); ++d)
        if (channelSize[d] != 1 && p[d] != g[d])
          return false;
      return true;
    };

    SmallVector<std::pair<SiteRecord, SiteRecord>> edges;
    for (auto &p : puts[name])
      for (auto &g : gets[name])
        if (matches(p.idx, g.idx))
          edges.push_back({p, g});

    if (edges.empty())
      return chanOp.emitOpError(
          "no matching put/get pairs found while enumerating tiles");

    // Try multicast: dx==0 for every edge and dy forms {1..n} or {-n..-1}
    // (or the symmetric case with dx varying and dy==0).
    auto classifyAxis = [&](bool xVaries) -> std::optional<ChannelInfo> {
      std::set<int64_t> varying, fixed;
      for (auto &[p, g] : edges) {
        int64_t d = xVaries ? (g.gx - p.gx) : (g.gy - p.gy);
        int64_t f = xVaries ? (g.gy - p.gy) : (g.gx - p.gx);
        varying.insert(d);
        fixed.insert(f);
      }
      if (fixed.size() != 1 || *fixed.begin() != 0)
        return std::nullopt;
      bool ascending = *varying.begin() >= 1;
      bool descending = *varying.rbegin() <= -1;
      if (ascending && !descending) {
        int64_t n = *varying.rbegin();
        if (n < 2)
          return std::nullopt; // a single-hop edge is a unit chain, not a
                                // (degenerate) multicast.
        for (int64_t i = 1; i <= n; ++i)
          if (!varying.count(i))
            return std::nullopt;
        if ((int64_t)varying.size() != n)
          return std::nullopt;
        ChannelInfo ci;
        ci.kind = ChannelInfo::Multicast;
        ci.axis = xVaries ? 0 : 1;
        ci.sign = 1;
        ci.n = n;
        return ci;
      }
      if (descending && !ascending) {
        int64_t n = -*varying.begin();
        if (n < 2)
          return std::nullopt; // a single-hop edge is a unit chain, not a
                                // (degenerate) multicast.
        for (int64_t i = 1; i <= n; ++i)
          if (!varying.count(-i))
            return std::nullopt;
        if ((int64_t)varying.size() != n)
          return std::nullopt;
        ChannelInfo ci;
        ci.kind = ChannelInfo::Multicast;
        ci.axis = xVaries ? 0 : 1;
        ci.sign = -1;
        ci.n = n;
        return ci;
      }
      return std::nullopt;
    };

    std::optional<ChannelInfo> ci = classifyAxis(/*xVaries=*/false);
    if (!ci)
      ci = classifyAxis(/*xVaries=*/true);

    if (!ci) {
      // Ring / chain along one axis (spec v2 §4). Every edge steps along a
      // single, shared axis; short (unit) edges use the parity streams and
      // any longer edges (the wrap-around) share a second direction and use
      // a wrap stream. A pure chain with no long edge reproduces v1's
      // unit-chain behavior.
      bool axisIsX = false, axisKnown = false, axisOk = true;
      for (auto &[p, g] : edges) {
        int64_t ddx = g.gx - p.gx, ddy = g.gy - p.gy;
        bool xNZ = ddx != 0, yNZ = ddy != 0;
        if (xNZ == yNZ) { // both zero (self) or both nonzero (diagonal)
          axisOk = false;
          break;
        }
        if (!axisKnown) {
          axisIsX = xNZ;
          axisKnown = true;
        } else if (axisIsX != xNZ) {
          axisOk = false;
          break;
        }
      }
      if (axisOk && axisKnown) {
        std::map<std::pair<int64_t, int64_t>, int> outDeg, inDeg;
        std::set<int64_t> deltas;
        for (auto &[p, g] : edges) {
          int64_t delta = axisIsX ? (g.gx - p.gx) : (g.gy - p.gy);
          outDeg[{p.gx, p.gy}]++;
          inDeg[{g.gx, g.gy}]++;
          deltas.insert(delta);
        }
        bool degreeOk = true;
        for (auto &kv : outDeg)
          degreeOk &= kv.second <= 1;
        for (auto &kv : inDeg)
          degreeOk &= kv.second <= 1;
        // Base edge = the shortest delta (any length: unit chains, but also
        // the 2^s-hop edges of a tree reduction); an optional second delta is
        // the wrap-around.
        if (degreeOk && !deltas.empty() && deltas.size() <= 2) {
          ChannelInfo c;
          c.kind = ChannelInfo::Ring;
          int64_t d = *deltas.begin();
          for (int64_t cand : deltas)
            if (std::abs(cand) < std::abs(d))
              d = cand;
          c.dx = axisIsX ? d : 0;
          c.dy = axisIsX ? 0 : d;
          c.baseHops = std::abs(d);
          c.baseStepX = axisIsX ? (d > 0 ? 1 : -1) : 0;
          c.baseStepY = axisIsX ? 0 : (d > 0 ? 1 : -1);
          if (deltas.size() == 2) {
            int64_t L = *deltas.begin() == d ? *std::next(deltas.begin())
                                             : *deltas.begin();
            c.hasWrap = true;
            c.wrapDx = axisIsX ? L : 0;
            c.wrapDy = axisIsX ? 0 : L;
            int64_t step = L > 0 ? 1 : -1;
            c.wrapStepX = axisIsX ? step : 0;
            c.wrapStepY = axisIsX ? 0 : step;
            c.wrapHops = std::abs(L);
          }
          for (auto &[p, g] : edges) {
            int64_t delta = axisIsX ? (g.gx - p.gx) : (g.gy - p.gy);
            int64_t senderCoord = axisIsX ? p.gx : p.gy;
            // Parity of the sender's position in units of the base edge
            // length keeps consecutive relays on distinct colours.
            int sel = delta == d ? (mod2(senderCoord / c.baseHops) == 0 ? 0 : 1)
                                 : 2;
            c.outStream[{p.gx, p.gy}] = sel;
            c.inStream[{g.gx, g.gy}] = sel;
          }
          ci = c;
        }
      }
    }

    if (!ci)
      return chanOp.emitOpError(
          "unsupported communication pattern for channel @")
             << name;

    auto elemTy = spadaScalarType(chanOp, elemTypeOf[name]);
    if (failed(elemTy))
      return failure();
    ci->elemType = *elemTy;

    if (ci->kind == ChannelInfo::Multicast) {
      ci->channelNum = nextChannel++;
    } else {
      ci->evenChannelNum = nextChannel++;
      ci->oddChannelNum = nextChannel++;
      if (ci->hasWrap)
        ci->wrapChannelNum = nextChannel++;
    }
    if (nextChannel > 21)
      return chanOp.emitOpError(
          "kernel needs more than the 21 hardware/virtual channels SpaDA "
          "reserves for non-memcpy streams");

    result[name] = *ci;
  }
  return result;
}

//===----------------------------------------------------------------------===//
// Role key construction (spec 3.4.1).
//===----------------------------------------------------------------------===//

LogicalResult SpadaEmitter::buildRoleKey(Block &block, Env &env,
                                          HerdInfo &herd, int64_t lx,
                                          int64_t ly, ChannelMap &channels,
                                          std::string &key) {
  for (Operation &op : block) {
    if (auto ifOp = dyn_cast<affine::AffineIfOp>(op)) {
      auto holds = evalAffineIf(ifOp, env);
      if (!holds)
        return ifOp.emitOpError("condition does not fold to a constant");
      key += (*holds ? 'T' : 'F');
      if (*holds) {
        if (failed(buildRoleKey(*ifOp.getThenBlock(), env, herd, lx, ly,
                                 channels, key)))
          return failure();
      } else if (ifOp.hasElse()) {
        if (failed(buildRoleKey(*ifOp.getElseBlock(), env, herd, lx, ly,
                                 channels, key)))
          return failure();
      }
      continue;
    }
    if (auto forOp = dyn_cast<affine::AffineForOp>(op)) {
      if (failed(buildRoleKey(*forOp.getBody(), env, herd, lx, ly, channels,
                               key)))
        return failure();
      continue;
    }
    if (auto put = dyn_cast<air::ChannelPutOp>(op)) {
      StringRef name = put.getChanName();
      auto it = channels.find(name);
      if (it != channels.end() && it->second.kind == ChannelInfo::Ring) {
        auto sel = lookupRingStream(put, it->second.outStream, herd.xloc + lx,
                                     herd.yloc + ly, "out");
        if (failed(sel))
          return failure();
        key += ("P" + name + ringStreamSuffix(*sel)).str();
      } else {
        key += ("p" + name).str();
      }
      continue;
    }
    if (auto get = dyn_cast<air::ChannelGetOp>(op)) {
      StringRef name = get.getChanName();
      auto it = channels.find(name);
      if (it != channels.end() && it->second.kind == ChannelInfo::Ring) {
        auto sel = lookupRingStream(get, it->second.inStream, herd.xloc + lx,
                                     herd.yloc + ly, "in");
        if (failed(sel))
          return failure();
        key += ("G" + name + ringStreamSuffix(*sel)).str();
      } else {
        key += ("g" + name).str();
      }
      continue;
    }
  }
  return success();
}

//===----------------------------------------------------------------------===//
// Expression printing.
//===----------------------------------------------------------------------===//

FailureOr<std::string>
SpadaEmitter::printAffineExpr(AffineExpr e, ValueRange operands,
                               HerdInfo &herd, int64_t lx, int64_t ly,
                               DenseMap<Value, std::string> &loopVarNames) {
  switch (e.getKind()) {
  case AffineExprKind::Constant:
    return std::to_string(cast<AffineConstantExpr>(e).getValue());
  case AffineExprKind::DimId:
  case AffineExprKind::SymbolId: {
    unsigned pos = e.getKind() == AffineExprKind::DimId
                       ? cast<AffineDimExpr>(e).getPosition()
                       : cast<AffineSymbolExpr>(e).getPosition();
    return printExpr(operands[pos], herd, lx, ly, loopVarNames);
  }
  case AffineExprKind::Add: {
    auto bin = cast<AffineBinaryOpExpr>(e);
    // Detect `a + (-1 * b)` as `a - b` for nicer output.
    if (auto rhsMul = dyn_cast<AffineBinaryOpExpr>(bin.getRHS());
        rhsMul && rhsMul.getKind() == AffineExprKind::Mul) {
      if (auto c = dyn_cast<AffineConstantExpr>(rhsMul.getRHS());
          c && c.getValue() == -1) {
        auto l = printAffineExpr(bin.getLHS(), operands, herd, lx, ly,
                                  loopVarNames);
        auto r = printAffineExpr(rhsMul.getLHS(), operands, herd, lx, ly,
                                  loopVarNames);
        if (failed(l) || failed(r))
          return failure();
        return ("(" + *l + " - " + *r + ")");
      }
    }
    auto l = printAffineExpr(bin.getLHS(), operands, herd, lx, ly,
                              loopVarNames);
    auto r = printAffineExpr(bin.getRHS(), operands, herd, lx, ly,
                              loopVarNames);
    if (failed(l) || failed(r))
      return failure();
    return ("(" + *l + " + " + *r + ")");
  }
  case AffineExprKind::Mul: {
    auto bin = cast<AffineBinaryOpExpr>(e);
    auto l = printAffineExpr(bin.getLHS(), operands, herd, lx, ly,
                              loopVarNames);
    auto r = printAffineExpr(bin.getRHS(), operands, herd, lx, ly,
                              loopVarNames);
    if (failed(l) || failed(r))
      return failure();
    return ("(" + *l + " * " + *r + ")");
  }
  case AffineExprKind::Mod:
  case AffineExprKind::FloorDiv:
  case AffineExprKind::CeilDiv: {
    auto bin = cast<AffineBinaryOpExpr>(e);
    auto l = printAffineExpr(bin.getLHS(), operands, herd, lx, ly,
                              loopVarNames);
    auto r = printAffineExpr(bin.getRHS(), operands, herd, lx, ly,
                              loopVarNames);
    if (failed(l) || failed(r))
      return failure();
    const char *op = e.getKind() == AffineExprKind::Mod ? " % " : " / ";
    return ("(" + *l + op + *r + ")");
  }
  }
  return failure();
}

FailureOr<std::string>
SpadaEmitter::printExpr(Value v, HerdInfo &herd, int64_t lx, int64_t ly,
                         DenseMap<Value, std::string> &loopVarNames) {
  if (auto it = loopVarNames.find(v); it != loopVarNames.end())
    return it->second;
  auto ids = herd.op.getIds();
  if (v == ids[0])
    return herd.xloc == 0 ? std::string("x")
                           : ("(x - " + std::to_string(herd.xloc) + ")");
  if (ids.size() > 1 && v == ids[1])
    return herd.yloc == 0 ? std::string("y")
                           : ("(y - " + std::to_string(herd.yloc) + ")");

  Operation *def = v.getDefiningOp();
  if (!def)
    return v.getParentBlock()->getParentOp()->emitError(
        "unsupported value with no defining op in a SpaDA expression");

  if (auto c = dyn_cast<arith::ConstantOp>(def)) {
    if (auto f = dyn_cast<FloatAttr>(c.getValue()))
      return fmtFloat(f.getValue());
    if (auto i = dyn_cast<IntegerAttr>(c.getValue()))
      return std::to_string(i.getInt());
    return def->emitError("unsupported constant type in a SpaDA expression");
  }
  if (auto cast = dyn_cast<arith::IndexCastOp>(def))
    return printExpr(cast.getIn(), herd, lx, ly, loopVarNames);

  auto binary = [&](Value lhs, Value rhs,
                     const char *op) -> FailureOr<std::string> {
    auto l = printExpr(lhs, herd, lx, ly, loopVarNames);
    auto r = printExpr(rhs, herd, lx, ly, loopVarNames);
    if (failed(l) || failed(r))
      return failure();
    return ("(" + *l + " " + op + " " + *r + ")");
  };
  if (auto op = dyn_cast<arith::AddFOp>(def))
    return binary(op.getLhs(), op.getRhs(), "+");
  if (auto op = dyn_cast<arith::SubFOp>(def))
    return binary(op.getLhs(), op.getRhs(), "-");
  if (auto op = dyn_cast<arith::MulFOp>(def))
    return binary(op.getLhs(), op.getRhs(), "*");
  if (auto op = dyn_cast<arith::DivFOp>(def))
    return binary(op.getLhs(), op.getRhs(), "/");
  if (auto op = dyn_cast<arith::AddIOp>(def))
    return binary(op.getLhs(), op.getRhs(), "+");
  if (auto op = dyn_cast<arith::SubIOp>(def))
    return binary(op.getLhs(), op.getRhs(), "-");
  if (auto op = dyn_cast<arith::MulIOp>(def))
    return binary(op.getLhs(), op.getRhs(), "*");

  if (auto apply = dyn_cast<affine::AffineApplyOp>(def)) {
    AffineMap map = apply.getAffineMap();
    return printAffineExpr(map.getResult(0), apply.getMapOperands(), herd, lx,
                            ly, loopVarNames);
  }

  std::string bufName;
  if (auto load = dyn_cast<affine::AffineLoadOp>(def)) {
    auto idx = flattenAccess(def, load.getMemRef(), {}, load.getAffineMap(),
                              load.getMapOperands(), herd, lx, ly,
                              loopVarNames, bufName);
    if (failed(idx))
      return failure();
    return (bufName + "[" + *idx + "]");
  }
  if (auto load = dyn_cast<memref::LoadOp>(def)) {
    SmallVector<Value> indices(load.getIndices());
    auto idx = flattenAccess(def, load.getMemRef(), indices, std::nullopt, {},
                              herd, lx, ly, loopVarNames, bufName);
    if (failed(idx))
      return failure();
    return (bufName + "[" + *idx + "]");
  }

  return def->emitError("unsupported operation in a SpaDA expression: ")
         << def->getName();
}

FailureOr<std::string> SpadaEmitter::flattenAccess(
    Operation *op, Value memref, ArrayRef<Value> plainIndices,
    std::optional<AffineMap> map, ValueRange mapOperands, HerdInfo &herd,
    int64_t lx, int64_t ly, DenseMap<Value, std::string> &loopVarNames,
    std::string &bufName) {
  bufName = lookupL1Name(herd, memref);
  if (bufName.empty())
    return op->emitError(
        "load/store target is not a herd-local (L1) buffer; only L1 buffers "
        "may be loaded/stored directly in a herd body");
  auto ty = cast<MemRefType>(memref.getType());
  ArrayRef<int64_t> shape = ty.getShape();
  unsigned rank = shape.size();

  // Evaluate each dimension's index expression once, then either join them
  // (spec v2 §1: a ranked, i.e. non-flat, buffer keeps its memref rank and
  // is indexed with one expression per dimension) or flatten them
  // row-major (a flat buffer: only a channel-get destination is flat).
  SmallVector<std::string> dims(rank);
  for (unsigned d = 0; d < rank; ++d) {
    FailureOr<std::string> idxStr;
    if (map) {
      idxStr = printAffineExpr(map->getResult(d), mapOperands, herd, lx, ly,
                                loopVarNames);
    } else {
      idxStr = printExpr(plainIndices[d], herd, lx, ly, loopVarNames);
    }
    if (failed(idxStr))
      return failure();
    dims[d] = *idxStr;
  }

  bool flat = herd.l1IsFlat.lookup(memref);
  if (!flat) {
    std::string result;
    for (unsigned d = 0; d < rank; ++d) {
      if (d)
        result += ", ";
      result += dims[d];
    }
    return result;
  }

  SmallVector<int64_t> strides(rank, 1);
  for (int64_t d = (int64_t)rank - 2; d >= 0; --d)
    strides[d] = strides[d + 1] * shape[d + 1];
  std::string result;
  for (unsigned d = 0; d < rank; ++d) {
    if (d)
      result += " + ";
    result += strides[d] == 1 ? dims[d]
                               : (dims[d] + " * " + std::to_string(strides[d]));
  }
  return result;
}

//===----------------------------------------------------------------------===//
// Herd body code generation.
//===----------------------------------------------------------------------===//

static const char *loopVarName(int depth) {
  static const char *names[] = {"k", "l", "m", "n", "o",
                                 "p", "q", "r", "s", "t"};
  if (depth < (int)(sizeof(names) / sizeof(names[0])))
    return names[depth];
  return "k";
}

LogicalResult SpadaEmitter::emitHerdBody(
    Block &block, HerdInfo &herd, Env &env, int64_t lx, int64_t ly,
    ChannelMap &channels, DenseMap<Value, std::string> &loopVarNames,
    int depth, raw_ostream &o, DenseMap<Value, CompletionRecord> &completions,
    int &nextCompletionId) {
  std::string ind((depth + 3) * 2, ' ');
  for (Operation &op : block) {
    if (isa<memref::AllocOp, memref::DeallocOp, air::HerdTerminatorOp,
            affine::AffineYieldOp>(op))
      continue;
    if (isa<affine::AffineApplyOp, arith::ConstantOp, arith::AddFOp,
            arith::SubFOp, arith::MulFOp, arith::DivFOp, arith::AddIOp,
            arith::SubIOp, arith::MulIOp, arith::IndexCastOp,
            affine::AffineLoadOp, memref::LoadOp>(op))
      continue; // pure producers, resolved lazily by printExpr.

    if (auto ifOp = dyn_cast<affine::AffineIfOp>(op)) {
      auto holds = evalAffineIf(ifOp, env);
      if (!holds)
        return ifOp.emitOpError("condition does not fold to a constant");
      if (*holds) {
        if (failed(emitHerdBody(*ifOp.getThenBlock(), herd, env, lx, ly,
                                 channels, loopVarNames, depth, o,
                                 completions, nextCompletionId)))
          return failure();
      } else if (ifOp.hasElse()) {
        if (failed(emitHerdBody(*ifOp.getElseBlock(), herd, env, lx, ly,
                                 channels, loopVarNames, depth, o,
                                 completions, nextCompletionId)))
          return failure();
      }
      continue;
    }

    if (auto forOp = dyn_cast<affine::AffineForOp>(op)) {
      if (!forOp.hasConstantBounds() || forOp.getStepAsInt() != 1)
        return forOp.emitOpError(
            "expected constant bounds and step 1 for a SpaDA for loop");
      int64_t lo = forOp.getConstantLowerBound();
      int64_t hi = forOp.getConstantUpperBound();
      const char *name = loopVarName(depth);
      loopVarNames[forOp.getInductionVar()] = name;
      o << ind << "for i16 " << name << " in [" << lo << ":" << hi << "] {\n";
      if (failed(emitHerdBody(*forOp.getBody(), herd, env, lx, ly, channels,
                               loopVarNames, depth + 1, o, completions,
                               nextCompletionId)))
        return failure();
      o << ind << "}\n";
      continue;
    }

    if (auto store = dyn_cast<affine::AffineStoreOp>(op)) {
      std::string bufName;
      auto idx = flattenAccess(store, store.getMemRef(), {},
                                store.getAffineMap(), store.getMapOperands(),
                                herd, lx, ly, loopVarNames, bufName);
      if (failed(idx))
        return failure();
      if (herd.l1IsFlat.lookup(store.getMemRef()) &&
          idx->find(' ') != std::string::npos)
        return store.emitOpError(
            "SpaDA cannot store through a computed index; only channel-get "
            "destinations are flat and they must be read, not written, "
            "inside loops");
      auto rhs = printExpr(store.getValueToStore(), herd, lx, ly, loopVarNames);
      if (failed(rhs))
        return failure();
      o << ind << bufName << "[" << *idx << "] = " << *rhs << "\n";
      continue;
    }
    if (auto store = dyn_cast<memref::StoreOp>(op)) {
      std::string bufName;
      SmallVector<Value> indices(store.getIndices());
      auto idx = flattenAccess(store, store.getMemRef(), indices,
                                std::nullopt, {}, herd, lx, ly, loopVarNames,
                                bufName);
      if (failed(idx))
        return failure();
      if (herd.l1IsFlat.lookup(store.getMemRef()) &&
          idx->find(' ') != std::string::npos)
        return store.emitOpError(
            "SpaDA cannot store through a computed index; only channel-get "
            "destinations are flat and they must be read, not written, "
            "inside loops");
      auto rhs =
          printExpr(store.getValueToStore(), herd, lx, ly, loopVarNames);
      if (failed(rhs))
        return failure();
      o << ind << bufName << "[" << *idx << "] = " << *rhs << "\n";
      continue;
    }

    if (auto dma = dyn_cast<air::DmaMemcpyNdOp>(op)) {
      Value src = dma.getSrcMemref(), dst = dma.getDstMemref();
      Value l2ArgDst = getL2AllocOfHerdArg(herd.op, dst);
      Value l2ArgSrc = getL2AllocOfHerdArg(herd.op, src);
      bool dstIsL2 = (bool)l2ArgDst, srcIsL2 = (bool)l2ArgSrc;
      if (dstIsL2 == srcIsL2)
        return dma.emitOpError(
            "expected exactly one side of a herd-body dma_memcpy_nd to be "
            "a partitioned L2 herd argument and the other an L1 buffer");
      Value l2Alloc = dstIsL2 ? l2ArgDst : l2ArgSrc;
      auto it = l2Index.find(l2Alloc);
      if (it == l2Index.end())
        return dma.emitOpError("could not resolve the partitioned L2 buffer");
      L2AllocInfo &l2 = l2Allocs[it->second];
      std::string l1Name = lookupL1Name(herd, dstIsL2 ? src : dst);
      if (l1Name.empty())
        return dma.emitOpError(
            "the non-L2 side of a herd-body dma_memcpy_nd must be an L1 "
            "buffer");
      SmallVector<OpFoldResult> l2Sizes =
          dstIsL2 ? dma.getMixedDstSizes() : dma.getMixedSrcSizes();
      if (!l2Sizes.empty()) {
        int64_t prod = 1;
        for (auto s : l2Sizes) {
          if (auto attr = dyn_cast<Attribute>(s)) {
            prod *= cast<IntegerAttr>(attr).getInt();
          } else {
            auto c = evalIndex(cast<Value>(s), env);
            if (!c)
              return dma.emitOpError(
                  "dma size does not fold to a constant under the tile "
                  "environment");
            prod *= *c;
          }
        }
        if (prod != l2.blocksize)
          return dma.emitOpError("partial-block access not supported yet");
      }
      // Spec v2 §1: L2 arrays are declared with the partition block shape,
      // and every non-flat buffer keeps its own memref rank; build a loop
      // nest over the block's non-unit dimensions (a size-1 dimension is
      // always index 0, so it needs no loop variable of its own) and index
      // each side per-dimension (or, for a flat L1 buffer, flattened).
      ArrayRef<int64_t> blockShape = l2.blockShape;
      SmallVector<unsigned> nuPos;
      SmallVector<int64_t> nuSize;
      for (unsigned d = 0; d < blockShape.size(); ++d)
        if (blockShape[d] != 1) {
          nuPos.push_back(d);
          nuSize.push_back(blockShape[d]);
        }
      Value l1Val = dstIsL2 ? src : dst;
      bool l1Flat = herd.l1IsFlat.lookup(l1Val);
      ArrayRef<int64_t> l1Shape = cast<MemRefType>(l1Val.getType()).getShape();
      if (!l1Flat && (l1Shape.size() != nuSize.size() ||
                      !std::equal(l1Shape.begin(), l1Shape.end(),
                                  nuSize.begin())))
        return dma.emitOpError(
            "partitioned L2 block shape does not match the L1 buffer shape "
            "for a whole-block copy");

      SmallVector<std::string> loopVars(nuSize.size());
      std::string curInd = ind;
      for (size_t i = 0; i < nuSize.size(); ++i) {
        loopVars[i] = loopVarName(depth + (int)i);
        o << curInd << "for i16 " << loopVars[i] << " in [0:" << nuSize[i]
          << "] {\n";
        curInd += "  ";
      }
      std::string l2Access;
      {
        size_t c = 0;
        for (unsigned d = 0; d < blockShape.size(); ++d) {
          if (d)
            l2Access += ", ";
          l2Access += blockShape[d] == 1 ? "0" : loopVars[c++];
        }
      }
      std::string l1Access;
      if (!l1Flat) {
        for (size_t i = 0; i < loopVars.size(); ++i) {
          if (i)
            l1Access += ", ";
          l1Access += loopVars[i];
        }
        if (l1Access.empty())
          l1Access = "0";
      } else {
        if (loopVars.empty()) {
          l1Access = "0";
        } else {
          SmallVector<int64_t> strides(l1Shape.size(), 1);
          for (int64_t d = (int64_t)l1Shape.size() - 2; d >= 0; --d)
            strides[d] = strides[d + 1] * l1Shape[d + 1];
          if (loopVars.size() != l1Shape.size())
            return dma.emitOpError(
                "cannot derive a flat index for the L1 side of a "
                "whole-block copy: rank mismatch with the L2 block shape");
          for (size_t d = 0; d < l1Shape.size(); ++d) {
            if (d)
              l1Access += " + ";
            l1Access += strides[d] == 1
                            ? loopVars[d]
                            : (loopVars[d] + " * " + std::to_string(strides[d]));
          }
        }
      }
      if (l2Access.empty())
        l2Access = "0";
      if (dstIsL2)
        o << curInd << l2.name << "[" << l2Access << "] = " << l1Name << "["
          << l1Access << "]\n";
      else
        o << curInd << l1Name << "[" << l1Access << "] = " << l2.name << "["
          << l2Access << "]\n";
      for (size_t i = nuSize.size(); i-- > 0;) {
        curInd.resize(curInd.size() - 2);
        o << curInd << "}\n";
      }
      continue;
    }

    if (auto put = dyn_cast<air::ChannelPutOp>(op)) {
      std::string bufName = lookupL1Name(herd, put.getSrc());
      if (bufName.empty())
        return put.emitOpError("channel put source must be an L1 buffer");
      auto it = channels.find(put.getChanName());
      if (it == channels.end())
        return put.emitOpError("channel was not classified");
      const ChannelInfo &ci = it->second;
      std::string streamName;
      if (ci.kind == ChannelInfo::Multicast) {
        streamName = put.getChanName().str();
      } else {
        auto sel = lookupRingStream(put, ci.outStream, herd.xloc + lx,
                                     herd.yloc + ly, "out");
        if (failed(sel))
          return failure();
        streamName =
            (put.getChanName() + "_" + ringStreamSuffix(*sel)).str();
      }
      // Spec v2 §2: an async put becomes a completion; a sync one keeps
      // v1's blocking `await send`.
      if (Value tok = put.getAsyncToken()) {
        std::string cname = "c" + std::to_string(nextCompletionId++);
        completions[tok] = {cname, depth};
        o << ind << "completion " << cname << " = send(" << bufName << ", "
          << streamName << ")\n";
      } else {
        o << ind << "await send(" << bufName << ", " << streamName << ")\n";
      }
      continue;
    }
    if (auto get = dyn_cast<air::ChannelGetOp>(op)) {
      std::string bufName = lookupL1Name(herd, get.getDst());
      if (bufName.empty())
        return get.emitOpError("channel get destination must be an L1 "
                                "buffer");
      auto it = channels.find(get.getChanName());
      if (it == channels.end())
        return get.emitOpError("channel was not classified");
      const ChannelInfo &ci = it->second;
      std::string streamName;
      if (ci.kind == ChannelInfo::Multicast) {
        streamName = get.getChanName().str();
      } else {
        auto sel = lookupRingStream(get, ci.inStream, herd.xloc + lx,
                                     herd.yloc + ly, "in");
        if (failed(sel))
          return failure();
        streamName =
            (get.getChanName() + "_" + ringStreamSuffix(*sel)).str();
      }
      // Spec v2 §2: an async get also becomes a completion; a sync one
      // keeps v1's blocking `await receive`.
      if (Value tok = get.getAsyncToken()) {
        std::string cname = "c" + std::to_string(nextCompletionId++);
        completions[tok] = {cname, depth};
        o << ind << "completion " << cname << " = receive(" << bufName
          << ", " << streamName << ")\n";
      } else {
        o << ind << "await receive(" << bufName << ", " << streamName
          << ")\n";
      }
      continue;
    }

    if (auto waitAll = dyn_cast<air::WaitAllOp>(op)) {
      // Spec v2 §2: `air.wait_all` with a result is not needed for v2's
      // input contract (Cannon's wait_all has no result); anything else
      // that produces a token and feeds it to further users is an error.
      if (waitAll.getAsyncToken())
        return waitAll.emitOpError(
            "air.wait_all with a result is not supported; v2 only lowers "
            "the no-result form");
      for (Value dep : waitAll.getAsyncDependencies()) {
        auto it = completions.find(dep);
        if (it == completions.end())
          continue; // not a channel completion; ignored per spec v2 §2.
        if (it->second.depth != depth)
          return waitAll.emitOpError(
              "a completion's token escaped the loop/block it was declared "
              "in; SpaDA requires the await in the same body");
        o << ind << "await " << it->second.name << "\n";
      }
      continue;
    }

    return op.emitError("unsupported operation in a SpaDA herd body: ")
           << op.getName();
  }
  return success();
}

//===----------------------------------------------------------------------===//
// Pass 2: emission.
//===----------------------------------------------------------------------===//

void SpadaEmitter::emitPlaceBlock(int64_t ux0, int64_t ux1, int64_t uy0,
                                   int64_t uy1) {
  os << "  place i16 x, i16 y in [" << fmtRect(ux0, ux1, uy0, uy1) << "] {\n";
  // Spec v2 §1: a partitioned L2 buffer is always declared with its
  // partition block shape (ranked).
  for (auto &l2 : l2Allocs)
    os << "    " << *spadaScalarType(l2.alloc.getDefiningOp(),
                                      l2.ty.getElementType())
       << "[" << formatDims(l2.blockShape) << "] " << l2.name << "\n";
  for (auto &herd : herds) {
    for (Value alloc : herd.l1Allocs) {
      auto ty = cast<MemRefType>(alloc.getType());
      bool flat = herd.l1IsFlat.lookup(alloc);
      std::string dims;
      if (flat) {
        int64_t size = 1;
        for (int64_t s : ty.getShape())
          size *= s;
        dims = std::to_string(size);
      } else {
        dims = formatDims(ty.getShape());
      }
      os << "    " << *spadaScalarType(alloc.getDefiningOp(),
                                        ty.getElementType())
         << "[" << dims << "] " << herd.l1Names.lookup(alloc) << "\n";
    }
  }
  os << "  }\n\n";
}

LogicalResult SpadaEmitter::emitSegDmaGroup(ArrayRef<unsigned> dmaIdxs) {
  os << "  phase {\n";
  for (unsigned idx : dmaIdxs) {
    SegDmaInfo &dma = segDmas[idx];
    L2AllocInfo &l2 = l2Allocs[dma.l2Idx];
    KernelArgInfo &arg = kernelArgs[dma.funcArgIdx];
    (void)arg;
    std::string ix = l2.ox0 == 0 ? "x" : ("x - " + std::to_string(l2.ox0));
    std::string iy = l2.oy0 == 0 ? "y" : ("y - " + std::to_string(l2.oy0));
    os << "    compute i16 x, i16 y in ["
       << fmtRect(l2.ox0, l2.ox1, l2.oy0, l2.oy1) << "] {\n";
    if (dma.dir == SegDmaDir::L3ToL2)
      os << "      await receive(" << l2.name << ", arg" << dma.funcArgIdx
         << "_in[" << ix << ", " << iy << "])\n";
    else
      os << "      await send(" << l2.name << ", arg" << dma.funcArgIdx
         << "_out[" << ix << ", " << iy << "])\n";
    os << "    }\n";
  }
  os << "  }\n\n";
  return success();
}

LogicalResult SpadaEmitter::emitHerdPhase(HerdInfo &herd) {
  SmallVector<std::string> channelOrder;
  auto channelsOr = classifyHerdChannels(herd, channelOrder);
  if (failed(channelsOr))
    return failure();
  ChannelMap channels = *channelsOr;

  // Role grouping. Use a vector of (key, tiles) plus a std::map index so
  // that roles are iterated in first-occurrence order (std::string is not a
  // valid llvm::DenseMap/MapVector key type).
  std::vector<std::pair<std::string, SmallVector<std::pair<int64_t, int64_t>>>>
      roleGroups;
  std::map<std::string, size_t> roleIndex;
  for (int64_t x = 0; x < herd.sx; ++x) {
    for (int64_t y = 0; y < herd.sy; ++y) {
      Env env;
      auto ids = herd.op.getIds();
      auto sizes = herd.op.getSize();
      env[ids[0]] = x;
      env[sizes[0]] = herd.sx;
      if (ids.size() > 1) {
        env[ids[1]] = y;
        env[sizes[1]] = herd.sy;
      }
      std::string key;
      if (failed(buildRoleKey(herd.op.getBody().front(), env, herd, x, y,
                               channels, key)))
        return failure();
      auto it = roleIndex.find(key);
      if (it == roleIndex.end()) {
        roleIndex[key] = roleGroups.size();
        roleGroups.push_back({key, {}});
        it = roleIndex.find(key);
      }
      roleGroups[it->second].second.push_back({x, y});
    }
  }

  os << "  phase {\n";
  if (!channelOrder.empty()) {
    os << "    dataflow i16 x, i16 y in ["
       << fmtRect(herd.xloc, herd.xloc + herd.sx, herd.yloc,
                   herd.yloc + herd.sy)
       << "] {\n";
    for (auto &name : channelOrder) {
      const ChannelInfo &ci = channels[name];
      if (ci.kind == ChannelInfo::Multicast) {
        std::string range =
            ci.sign > 0
                ? ("[1:" + std::to_string(ci.n + 1) + "]")
                : ("[-1:-" + std::to_string(ci.n + 1) + "]");
        std::string args = ci.axis == 1 ? ("0, " + range) : (range + ", 0");
        os << "      stream<" << ci.elemType << "> " << name
           << " = relative_stream(" << args << ") { hops = auto, channel = "
           << ci.channelNum << " }\n";
      } else {
        auto baseHops = [&](raw_ostream &o) {
          for (int64_t h = 0; h < ci.baseHops; ++h) {
            if (h)
              o << ", ";
            o << "(" << ci.baseStepX << ", " << ci.baseStepY << ")";
          }
        };
        os << "      stream<" << ci.elemType << "> " << name
           << "_even = relative_stream(" << ci.dx << ", " << ci.dy
           << ") { hops = [";
        baseHops(os);
        os << "], channel = " << ci.evenChannelNum << " }\n";
        os << "      stream<" << ci.elemType << "> " << name
           << "_odd = relative_stream(" << ci.dx << ", " << ci.dy
           << ") { hops = [";
        baseHops(os);
        os << "], channel = " << ci.oddChannelNum << " }\n";
        if (ci.hasWrap) {
          os << "      stream<" << ci.elemType << "> " << name
             << "_wrap = relative_stream(" << ci.wrapDx << ", " << ci.wrapDy
             << ") { hops = [";
          for (int64_t h = 0; h < ci.wrapHops; ++h) {
            if (h)
              os << ", ";
            os << "(" << ci.wrapStepX << ", " << ci.wrapStepY << ")";
          }
          os << "], channel = " << ci.wrapChannelNum << " }\n";
        }
      }
    }
    os << "    }\n";
  }

  for (auto &kv : roleGroups) {
    auto &tiles = kv.second;
    std::set<int64_t> xs, ys;
    for (auto &[x, y] : tiles) {
      xs.insert(x);
      ys.insert(y);
    }
    auto checkArith = [&](const std::set<int64_t> &s,
                           int64_t &lo, int64_t &hi,
                           int64_t &stride) -> LogicalResult {
      SmallVector<int64_t> v(s.begin(), s.end());
      lo = v.front();
      hi = v.back() + 1;
      stride = v.size() > 1 ? v[1] - v[0] : 1;
      for (size_t i = 1; i < v.size(); ++i)
        if (v[i] - v[i - 1] != stride)
          return herd.op.emitOpError(
              "a role group is not expressible as a strided rectangle");
      return success();
    };
    int64_t xlo, xhi, xstride, ylo, yhi, ystride;
    if (failed(checkArith(xs, xlo, xhi, xstride)) ||
        failed(checkArith(ys, ylo, yhi, ystride)))
      return failure();
    if ((int64_t)tiles.size() != (int64_t)xs.size() * (int64_t)ys.size())
      return herd.op.emitOpError(
          "a role group is not expressible as a strided rectangle");

    os << "    compute i16 x, i16 y in ["
       << fmtRect(herd.xloc + xlo, herd.xloc + xhi, herd.yloc + ylo,
                   herd.yloc + yhi, xstride, ystride)
       << "] {\n";
    auto [rx, ry] = tiles.front();
    Env env;
    auto ids = herd.op.getIds();
    auto sizes = herd.op.getSize();
    env[ids[0]] = rx;
    env[sizes[0]] = herd.sx;
    if (ids.size() > 1) {
      env[ids[1]] = ry;
      env[sizes[1]] = herd.sy;
    }
    DenseMap<Value, std::string> loopVarNames;
    // Spec v2 §2: completion ids (`cN`) are per compute-block.
    DenseMap<Value, CompletionRecord> completions;
    int nextCompletionId = 0;
    if (failed(emitHerdBody(herd.op.getBody().front(), herd, env, rx, ry,
                             channels, loopVarNames, /*depth=*/0, os,
                             completions, nextCompletionId)))
      return failure();
    os << "    }\n";
  }
  os << "  }\n\n";
  return success();
}

LogicalResult SpadaEmitter::emitAll() {
  // Kernel argument comments + signature.
  for (unsigned i = 0, e = funcOp.getNumArguments(); i < e; ++i) {
    if (!kernelArgs.count(i))
      continue;
    os << "// arg" << i << ": " << funcOp.getArgument(i).getType() << "\n";
  }

  SmallVector<std::string> argDecls;
  for (unsigned i = 0, e = funcOp.getNumArguments(); i < e; ++i) {
    auto it = kernelArgs.find(i);
    if (it == kernelArgs.end())
      continue;
    KernelArgInfo &info = it->second;
    auto elemTy = spadaScalarType(funcOp, info.ty.getElementType());
    if (failed(elemTy))
      return failure();
    if (info.hasIn) {
      L2AllocInfo &l2 = l2Allocs[info.l2InIdx];
      argDecls.push_back((Twine("stream<") + *elemTy + ", " +
                           Twine(l2.blocksize) + ">[" +
                           Twine(l2.ox1 - l2.ox0) + ", " +
                           Twine(l2.oy1 - l2.oy0) + "] readonly arg" +
                           Twine(i) + "_in")
                              .str());
    }
    if (info.hasOut) {
      L2AllocInfo &l2 = l2Allocs[info.l2OutIdx];
      argDecls.push_back((Twine("stream<") + *elemTy + ", " +
                           Twine(l2.blocksize) + ">[" +
                           Twine(l2.ox1 - l2.ox0) + ", " +
                           Twine(l2.oy1 - l2.oy0) + "] writeonly arg" +
                           Twine(i) + "_out")
                              .str());
    }
  }

  os << "kernel @" << funcOp.getName() << "<>(";
  for (size_t i = 0; i < argDecls.size(); ++i) {
    if (i)
      os << ",\n               ";
    os << argDecls[i];
  }
  os << ") {\n";

  // Union rectangle over all herds and all L2 owner rectangles.
  int64_t ux0 = INT64_MAX, ux1 = INT64_MIN, uy0 = INT64_MAX, uy1 = INT64_MIN;
  for (auto &h : herds) {
    ux0 = std::min(ux0, h.xloc);
    ux1 = std::max(ux1, h.xloc + h.sx);
    uy0 = std::min(uy0, h.yloc);
    uy1 = std::max(uy1, h.yloc + h.sy);
  }
  for (auto &l2 : l2Allocs) {
    ux0 = std::min(ux0, l2.ox0);
    ux1 = std::max(ux1, l2.ox1);
    uy0 = std::min(uy0, l2.oy0);
    uy1 = std::max(uy1, l2.oy1);
  }
  if (herds.empty() && l2Allocs.empty()) {
    ux0 = uy0 = 0;
    ux1 = uy1 = 1;
  }
  emitPlaceBlock(ux0, ux1, uy0, uy1);

  for (auto &item : program) {
    if (item.isHerd) {
      if (failed(emitHerdPhase(herds[item.herdIdx])))
        return failure();
    } else {
      if (failed(emitSegDmaGroup(item.dmas)))
        return failure();
    }
  }

  os << "}\n";
  return success();
}

LogicalResult SpadaEmitter::run() {
  if (failed(collectStructure()))
    return failure();
  return emitAll();
}

} // namespace

//===----------------------------------------------------------------------===//
// Registration.
//===----------------------------------------------------------------------===//

void xilinx::air::registerAIRToSpadaTranslation() {
  TranslateFromMLIRRegistration registration(
      "air-to-spada", "Emit a SpaDA Spatial IR kernel",
      [](ModuleOp module, raw_ostream &output) {
        SpadaEmitter emitter(module, output);
        return emitter.run();
      },
      [](DialectRegistry &registry) {
        registry.insert<air::airDialect, func::FuncDialect, arith::ArithDialect,
                         memref::MemRefDialect, scf::SCFDialect,
                         affine::AffineDialect>();
      });
}
