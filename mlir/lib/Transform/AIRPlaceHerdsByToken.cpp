//===- AIRPlaceHerdsByToken.cpp ---------------------------------*- C++ -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
//
// Materialises the affinity / concurrency semantics of AIRComputeModel.md
// §1.5 on a 2D tile fabric, and checks ownership of partitioned L2 memrefs.
//
//  * affinity class  : herds sharing an affinity token. Same shape, same
//                      (x_loc, y_loc); they execute one after the other on
//                      the same tiles, so data placed there stays resident.
//  * concurrency     : herds sharing a concurrency token, or connected by an
//                      air.channel (put in one herd, get in another), must be
//                      live at the same time and therefore get disjoint
//                      rectangles. Concurrency inside one affinity class is a
//                      deadlock and is reported.
//  * ownership       : inside a herd, every air.dma_memcpy_nd touching an L2
//                      memref that carries #air.partition must stay inside the
//                      block(s) owned by the executing tile. Enclosing
//                      affine.if conditions on the tile ids are honoured.
//
//===----------------------------------------------------------------------===//

#include "air/Transform/AIRPlaceHerdsByToken.h"
#include "air/Dialect/AIR/AIRDialect.h"
#include "air/Util/Util.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/EquivalenceClasses.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "air-place-herds-by-token"

using namespace mlir;
using namespace xilinx;

namespace {

//===----------------------------------------------------------------------===//
// Index evaluation under a per-tile environment
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

// True if `op` executes on the tile described by `env`, looking at every
// affine.if between `op` and `herd`.
static std::optional<bool> executesOnTile(Operation *op, air::HerdOp herd,
                                          const Env &env) {
  Operation *cur = op;
  while (cur && cur != herd.getOperation()) {
    Region *region = cur->getParentRegion();
    Operation *parent = cur->getParentOp();
    if (auto ifOp = dyn_cast_if_present<affine::AffineIfOp>(parent)) {
      auto holds = evalAffineIf(ifOp, env);
      if (!holds)
        return std::nullopt;
      bool inThen = region == &ifOp.getThenRegion();
      if (*holds != inThen)
        return false;
    }
    cur = parent;
  }
  return true;
}

//===----------------------------------------------------------------------===//
// Ownership check
//===----------------------------------------------------------------------===//

static air::PartitionAttr getPartition(Value memref) {
  auto alloc = memref.getDefiningOp<memref::AllocOp>();
  if (!alloc)
    return nullptr;
  return alloc->getAttrOfType<air::PartitionAttr>(
      air::PartitionAttr::getAttrName());
}

// Resolve a herd kernel argument to the partition attribute of the L2 alloc
// it is tied to (through segment args as well).
static air::PartitionAttr getPartitionOfHerdArg(air::HerdOp herd, Value v) {
  auto arg = dyn_cast<BlockArgument>(v);
  if (!arg || arg.getOwner()->getParentOp() != herd.getOperation())
    return nullptr;
  Value operand = herd.getTiedKernelOperand(arg);
  if (!operand)
    return nullptr;
  if (auto p = getPartition(operand))
    return p;
  // One more hop through an enclosing segment.
  if (auto segArg = dyn_cast<BlockArgument>(operand))
    if (auto seg = dyn_cast<air::SegmentOp>(segArg.getOwner()->getParentOp()))
      if (Value segOperand = seg.getTiedKernelOperand(segArg))
        return getPartition(segOperand);
  return nullptr;
}

// Enumerate the linear element indices of a dma access side.
static std::optional<SmallVector<int64_t>>
enumerateAccess(MemRefType ty, ArrayRef<OpFoldResult> offsets,
                ArrayRef<OpFoldResult> sizes, ArrayRef<OpFoldResult> strides,
                const Env &env) {
  auto shape = ty.getShape();
  int64_t total = 1;
  for (int64_t s : shape)
    total *= s;
  SmallVector<int64_t> out;
  if (offsets.empty()) {
    for (int64_t i = 0; i < total; ++i)
      out.push_back(i);
    return out;
  }
  auto evalOFR = [&](OpFoldResult ofr) -> std::optional<int64_t> {
    if (auto attr = dyn_cast<Attribute>(ofr))
      return cast<IntegerAttr>(attr).getInt();
    return evalIndex(cast<Value>(ofr), env);
  };
  SmallVector<int64_t> off, sz, st;
  for (auto o : offsets) {
    auto v = evalOFR(o);
    if (!v)
      return std::nullopt;
    off.push_back(*v);
  }
  for (auto o : sizes) {
    auto v = evalOFR(o);
    if (!v)
      return std::nullopt;
    sz.push_back(*v);
  }
  for (auto o : strides) {
    auto v = evalOFR(o);
    if (!v)
      return std::nullopt;
    st.push_back(*v);
  }
  // Base linear offset: offsets are coordinates in the memref when their
  // count equals the rank, else they are already linear (times stride).
  int64_t base = 0;
  if ((int64_t)off.size() == ty.getRank()) {
    int64_t mul = 1;
    for (int64_t d = ty.getRank() - 1; d >= 0; --d) {
      base += off[d] * mul;
      mul *= shape[d];
    }
  } else {
    for (size_t d = 0; d < off.size(); ++d)
      base += off[d] * st[d];
  }
  int64_t n = 1;
  for (int64_t s : sz)
    n *= s;
  if (n > (1 << 20))
    return std::nullopt;
  SmallVector<int64_t> idx(sz.size(), 0);
  for (int64_t i = 0; i < n; ++i) {
    int64_t lin = base;
    for (size_t d = 0; d < sz.size(); ++d)
      lin += idx[d] * st[d];
    out.push_back(lin);
    for (int64_t d = sz.size() - 1; d >= 0; --d) {
      if (++idx[d] < sz[d])
        break;
      idx[d] = 0;
    }
  }
  return out;
}

// Owner tile of the linear element `lin` of a memref under `part`.
static std::optional<SmallVector<int64_t>>
ownerOfElement(MemRefType ty, air::PartitionAttr part, int64_t lin) {
  auto shape = ty.getShape();
  auto block = part.getBlock();
  if ((int64_t)block.size() != ty.getRank())
    return std::nullopt;
  SmallVector<int64_t> coord(ty.getRank());
  for (int64_t d = ty.getRank() - 1; d >= 0; --d) {
    coord[d] = lin % shape[d];
    lin /= shape[d];
  }
  SmallVector<int64_t> blk(ty.getRank());
  for (int64_t d = 0; d < ty.getRank(); ++d)
    blk[d] = coord[d] / block[d];
  return part.getOwnerOfBlock(blk);
}

static LogicalResult checkOwnership(air::HerdOp herd,
                                    ArrayRef<int64_t> herdSizes) {
  LogicalResult result = success();
  herd.walk([&](air::DmaMemcpyNdOp dma) {
    for (int side = 0; side < 2; ++side) {
      Value memref = side == 0 ? dma.getSrcMemref() : dma.getDstMemref();
      air::PartitionAttr part = getPartitionOfHerdArg(herd, memref);
      if (!part)
        continue;
      auto ty = cast<MemRefType>(memref.getType());
      SmallVector<OpFoldResult> offsets =
          side == 0 ? dma.getMixedSrcOffsets() : dma.getMixedDstOffsets();
      SmallVector<OpFoldResult> sizes =
          side == 0 ? dma.getMixedSrcSizes() : dma.getMixedDstSizes();
      SmallVector<OpFoldResult> strides =
          side == 0 ? dma.getMixedSrcStrides() : dma.getMixedDstStrides();
      // Enumerate tiles.
      int64_t nTiles = 1;
      for (int64_t s : herdSizes)
        nTiles *= s;
      for (int64_t t = 0; t < nTiles; ++t) {
        SmallVector<int64_t> tile(herdSizes.size());
        int64_t rem = t;
        for (size_t d = 0; d < herdSizes.size(); ++d) {
          tile[d] = rem % herdSizes[d];
          rem /= herdSizes[d];
        }
        Env env;
        for (size_t d = 0; d < herdSizes.size(); ++d) {
          env[herd.getIds()[d]] = tile[d];
          env[herd.getSize()[d]] = herdSizes[d];
        }
        auto runs = executesOnTile(dma, herd, env);
        if (!runs) {
          dma.emitOpError("enclosing affine.if condition does not fold to a "
                          "constant for tile ")
              << "(" << tile[0] << ", " << (tile.size() > 1 ? tile[1] : 0)
              << ")";
          result = failure();
          return;
        }
        if (!*runs)
          continue;
        auto elems = enumerateAccess(ty, offsets, sizes, strides, env);
        if (!elems) {
          dma.emitOpError("access into partitioned memref cannot be evaluated "
                          "statically for tile ")
              << "(" << tile[0] << ", " << (tile.size() > 1 ? tile[1] : 0)
              << ")";
          result = failure();
          return;
        }
        for (int64_t lin : *elems) {
          auto owner = ownerOfElement(ty, part, lin);
          if (!owner || owner->size() != tile.size() ||
              !llvm::equal(*owner, tile)) {
            std::string ownerStr = "?";
            if (owner) {
              ownerStr = "(";
              for (size_t i = 0; i < owner->size(); ++i)
                ownerStr += (i ? ", " : "") + std::to_string((*owner)[i]);
              ownerStr += ")";
            }
            dma.emitOpError("tile (")
                << tile[0] << ", " << (tile.size() > 1 ? tile[1] : 0)
                << ") accesses element " << lin
                << " of a partitioned L2 memref owned by tile " << ownerStr
                << "; non-owner access must use an air.channel";
            result = failure();
            return;
          }
        }
      }
    }
  });
  return result;
}

//===----------------------------------------------------------------------===//
// Placement
//===----------------------------------------------------------------------===//

struct HerdInfo {
  air::HerdOp op;
  SmallVector<int64_t> sizes; // (x, y)
  unsigned cls = 0;
};

class AIRPlaceHerdsByToken
    : public air::impl::AIRPlaceHerdsByTokenBase<AIRPlaceHerdsByToken> {
public:
  AIRPlaceHerdsByToken() = default;
  AIRPlaceHerdsByToken(const AIRPlaceHerdsByToken &) {}
  AIRPlaceHerdsByToken(const air::AIRPlaceHerdsByTokenOptions &options)
      : AIRPlaceHerdsByTokenBase(options) {}

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<air::airDialect, affine::AffineDialect,
                    arith::ArithDialect, memref::MemRefDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    bool failed_ = false;
    module.walk([&](air::SegmentOp seg) {
      if (failed(processSegment(seg)))
        failed_ = true;
    });
    if (failed_)
      signalPassFailure();
  }

private:
  LogicalResult processSegment(air::SegmentOp seg) {
    SmallVector<HerdInfo> herds;
    for (auto herd : seg.getOps<air::HerdOp>()) {
      HerdInfo info{herd, {}, 0};
      for (Value s : herd.getSizeOperands()) {
        auto c = getConstantIntValue(s);
        if (!c)
          return herd.emitOpError("herd size must be a constant for placement");
        info.sizes.push_back(*c);
      }
      while (info.sizes.size() < 2)
        info.sizes.push_back(1);
      herds.push_back(info);
    }
    if (herds.empty())
      return success();

    // Affinity classes: union-find on shared affinity tokens.
    llvm::EquivalenceClasses<unsigned> ec;
    DenseMap<Value, unsigned> tokenRep;
    for (unsigned i = 0; i < herds.size(); ++i) {
      ec.insert(i);
      for (Value t : herds[i].op.getAffinityTokens()) {
        auto it = tokenRep.find(t);
        if (it == tokenRep.end())
          tokenRep[t] = i;
        else
          ec.unionSets(i, it->second);
      }
    }
    // Class ids in program order of their first member.
    DenseMap<unsigned, unsigned> leaderToCls;
    SmallVector<SmallVector<unsigned>> classes;
    for (unsigned i = 0; i < herds.size(); ++i) {
      unsigned leader = ec.getLeaderValue(i);
      auto it = leaderToCls.find(leader);
      if (it == leaderToCls.end()) {
        leaderToCls[leader] = classes.size();
        classes.push_back({i});
      } else {
        classes[it->second].push_back(i);
      }
      herds[i].cls = leaderToCls[leader];
    }
    // Same shape within a class.
    for (auto &members : classes)
      for (unsigned m : members)
        if (herds[m].sizes != herds[members[0]].sizes)
          return herds[m].op.emitOpError(
              "shares an affinity token with a herd of a different shape");

    // Concurrency edges: explicit tokens and channel put/get pairs.
    DenseMap<Value, SmallVector<unsigned>> concTokens;
    for (unsigned i = 0; i < herds.size(); ++i)
      for (Value t : herds[i].op.getConcurrencyTokens())
        concTokens[t].push_back(i);
    DenseMap<StringRef, SmallVector<unsigned>> putters, getters;
    for (unsigned i = 0; i < herds.size(); ++i) {
      herds[i].op.walk([&](air::ChannelPutOp p) {
        putters[p.getChanName()].push_back(i);
      });
      herds[i].op.walk([&](air::ChannelGetOp g) {
        getters[g.getChanName()].push_back(i);
      });
    }
    SmallVector<std::pair<unsigned, unsigned>> concurrent;
    for (auto &kv : concTokens)
      for (unsigned a : kv.second)
        for (unsigned b : kv.second)
          if (a < b)
            concurrent.push_back({a, b});
    for (auto &kv : putters)
      for (unsigned a : kv.second)
        for (unsigned b : getters.lookup(kv.first))
          if (a != b) {
            if (herds[a].cls == herds[b].cls)
              return herds[b].op.emitOpError("communicates through channel @")
                     << kv.first
                     << " with a herd in the same affinity class; "
                        "affinity-bound herds execute sequentially on the "
                        "same tiles, so this would deadlock";
            concurrent.push_back({std::min(a, b), std::max(a, b)});
          }
    for (auto [a, b] : concurrent)
      if (herds[a].cls == herds[b].cls)
        return herds[b].op.emitOpError(
            "shares both an affinity and a concurrency token with another "
            "herd; the two constraints are contradictory");

    // First-fit placement of classes on the fabric. All classes get
    // disjoint rectangles (a dependency-only relation could reuse tiles;
    // v1 does not exploit that).
    int64_t fx = clFabricX, fy = clFabricY;
    std::vector<std::vector<bool>> occ(fx, std::vector<bool>(fy, false));
    auto fits = [&](int64_t x0, int64_t y0, int64_t sx, int64_t sy) {
      if (x0 + sx > fx || y0 + sy > fy)
        return false;
      for (int64_t x = x0; x < x0 + sx; ++x)
        for (int64_t y = y0; y < y0 + sy; ++y)
          if (occ[x][y])
            return false;
      return true;
    };
    auto mark = [&](int64_t x0, int64_t y0, int64_t sx, int64_t sy) {
      for (int64_t x = x0; x < x0 + sx; ++x)
        for (int64_t y = y0; y < y0 + sy; ++y)
          occ[x][y] = true;
    };
    OpBuilder b(seg.getContext());
    for (auto &members : classes) {
      auto &first = herds[members[0]];
      int64_t sx = first.sizes[0], sy = first.sizes[1];
      // Honour a pre-existing explicit placement on any member.
      std::optional<std::pair<int64_t, int64_t>> fixed;
      for (unsigned m : members) {
        auto cx = herds[m].op.getColOffset();
        auto cy = herds[m].op.getRowOffset();
        if (cx && cy) {
          if (fixed && (fixed->first != (int64_t)*cx ||
                        fixed->second != (int64_t)*cy))
            return herds[m].op.emitOpError(
                "explicit x_loc/y_loc disagrees with another herd of its "
                "affinity class");
          fixed = {(int64_t)*cx, (int64_t)*cy};
        }
      }
      std::optional<std::pair<int64_t, int64_t>> loc = fixed;
      if (!loc) {
        for (int64_t y0 = 0; y0 < fy && !loc; ++y0)
          for (int64_t x0 = 0; x0 < fx && !loc; ++x0)
            if (fits(x0, y0, sx, sy))
              loc = {x0, y0};
      } else if (!fits(loc->first, loc->second, sx, sy)) {
        return first.op.emitOpError("explicit placement overlaps another "
                                    "concurrently live herd or the fabric");
      }
      if (!loc)
        return first.op.emitOpError("no free ")
               << sx << "x" << sy << " rectangle left on a " << fx << "x" << fy
               << " fabric";
      mark(loc->first, loc->second, sx, sy);
      for (unsigned m : members) {
        herds[m].op->setAttr(air::HerdOp::getColOffsetAttrName(),
                             b.getI64IntegerAttr(loc->first));
        herds[m].op->setAttr(air::HerdOp::getRowOffsetAttrName(),
                             b.getI64IntegerAttr(loc->second));
      }
    }

    if (clCheckOwnership)
      for (auto &h : herds)
        if (failed(checkOwnership(h.op, h.sizes)))
          return failure();
    return success();
  }
};

} // namespace

namespace xilinx {
namespace air {

std::unique_ptr<mlir::Pass> createAIRPlaceHerdsByTokenPass() {
  return std::make_unique<AIRPlaceHerdsByToken>();
}
std::unique_ptr<mlir::Pass>
createAIRPlaceHerdsByTokenPass(const AIRPlaceHerdsByTokenOptions &options) {
  return std::make_unique<AIRPlaceHerdsByToken>(options);
}

} // namespace air
} // namespace xilinx
