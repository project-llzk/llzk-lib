//===-- IncludeHelper.cpp - Helpers for LLZK file includes ------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains the implementation for file include helpers.
///
//===----------------------------------------------------------------------===//

#include "llzk/Dialect/Include/Util/IncludeHelper.h"

#include "llzk/Dialect/Include/IR/Ops.h"
#include "llzk/Util/ErrorHelper.h"
#include "llzk/Util/SymbolHelper.h"

#include <mlir/IR/AsmState.h>
#include <mlir/IR/Diagnostics.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OwningOpRef.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Parser/Parser.h>
#include <mlir/Support/LogicalResult.h>

#include <llvm/Support/Casting.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/SourceMgr.h>

#include <functional>

namespace llzk::include {

namespace {
using namespace mlir;

struct OpenFile {
  std::string resolvedPath;
  std::unique_ptr<llvm::MemoryBuffer> buffer;
};

inline FailureOr<OpenFile> openFile(EmitErrorFn emitError, const StringRef filename) {
  OpenFile r;

  auto buffer = GlobalSourceMgr::get().openIncludeFile(filename, r.resolvedPath);
  if (!buffer) {
    return emitError() << "could not find file \"" << filename << '"';
  }
  r.buffer = std::move(*buffer);
  return std::move(r);
}

FailureOr<OwningOpRef<ModuleOp>> parseFile(const StringRef filename, Operation *origin) {
  // Load raw contents of the file
  auto of = openFile(getEmitOpErrFn(origin), filename);
  if (failed(of)) {
    return failure();
  }

  // Parse the IR and write it in the destination block
  ParserConfig parseConfig(origin->getContext());
  llvm::StringRef contents = of->buffer->getBuffer();
  auto res = parseSourceString<ModuleOp>(contents, parseConfig, /*sourceName=*/of->resolvedPath);
  if (res) {
    return res;
  } else {
    return origin->emitOpError() << "could not parse file \"" << filename << '"';
  }
}

LogicalResult parseFile(const StringRef filename, Operation *origin, Block *container) {
  // Load raw contents of the file
  auto of = openFile(getEmitOpErrFn(origin), filename);
  if (failed(of)) {
    return failure();
  }

  // Parse the IR and write it in the destination block
  ParserConfig parseConfig(origin->getContext());
  llvm::StringRef contents = of->buffer->getBuffer();
  auto res = parseSourceString(contents, container, parseConfig, /*sourceName=*/of->resolvedPath);
  if (succeeded(res)) {
    return res;
  } else {
    return origin->emitOpError() << "could not parse file \"" << filename << '"';
  }
}

inline LogicalResult validateLoadedModuleOp(EmitErrorFn emitError, ModuleOp importedMod) {
  if (!importedMod->hasAttr(LANG_ATTR_NAME)) {
    return emitError()
        .append(
            "expected '", ModuleOp::getOperationName(), "' from included file to have \"",
            LANG_ATTR_NAME, "\" attribute"
        )
        .attachNote(importedMod.getLoc())
        .append("this should have \"", LANG_ATTR_NAME, "\" attribute");
  }
  if (importedMod.getSymNameAttr()) {
    return emitError()
        .append("expected '", ModuleOp::getOperationName(), "' from included file to be unnamed")
        .attachNote(importedMod.getLoc())
        .append("this should be unnamed");
  }
  return success();
}

/// Manages the inlining and the associated memory used.
/// It has a SQL-esque workflow. The operation can be commited if everything looks fine or it will
/// rollback when its lifetime ends unless it was commited.
class InlineOperationsGuard {
public:
  InlineOperationsGuard(MLIRContext *ctx, IncludeOp &tIncOp)
      : incOp(tIncOp), rewriter(ctx), dest(rewriter.createBlock(incOp->getBlock()->getParent())) {}

  ~InlineOperationsGuard() {
    if (commited) {
      // The container was inlined so get rid of the include op.
      rewriter.eraseOp(incOp);
    } else {
      // The container was not inlined so delete the container.
      dest->erase();
    }
  }

  /// Tells the guard that is safe to assume that the module was inserted into the destination
  void moduleWasLoaded() {
    assert(!dest->empty());
    blockWritten = true;
  }

  // Attempts to get the module written into the block
  FailureOr<ModuleOp> getModule() {
    // If the block is not ready return failure but do not emit diagnostics.
    if (!blockWritten) {
      return failure();
    }

    if (dest->empty()) {
      return incOp->emitOpError() << "failed to inline the module. No operation was written.";
    }

    auto &op = dest->front();
    if (!isa<ModuleOp>(op)) {
      return op.emitError()
          .append(
              "expected '", ModuleOp::getOperationName(),
              "' as top level operation of included file. Got '", op.getName(), "'."
          )
          .attachNote(incOp.getLoc())
          .append("from file included here");
    }
    return llvm::cast<ModuleOp>(op);
  }

  Block *getDest() { return dest; }

  FailureOr<ModuleOp> commit() {
    // Locate where to insert the inlined module
    rewriter.setInsertionPointAfter(incOp);
    auto insertionPoint = rewriter.getInsertionPoint();
    {
      // This op will be invalid after inlining the block
      auto modRes = getModule();
      // Won't commit on a failed result
      if (failed(modRes)) {
        return failure();
      }

      // Add the destination block after the insertion point.
      // dest becomes the source from which to move operations.
      rewriter.inlineBlockBefore(dest, rewriter.getInsertionBlock(), insertionPoint);
    }

    rewriter.setInsertionPointAfter(incOp);
    auto modOp = rewriter.getInsertionPoint();
    ModuleOp mod = llvm::dyn_cast<ModuleOp>(modOp);

    // Apply the name from the IncludeOp to the new ModuleOp
    mod.setSymNameAttr(incOp.getSymNameAttr());

    // All good so we mark as commited and return a reference to the newly generated module.
    commited = true;
    return mod;
  }

private:
  bool commited = false, blockWritten = false;
  IncludeOp &incOp;
  IRRewriter rewriter;
  Block *dest;
};
} // namespace

FailureOr<ModuleOp> IncludeOp::inlineAndErase() {
  InlineOperationsGuard guard(this->getContext(), *this);

  auto loadResult = parseFile(this->getPath(), *this, guard.getDest());
  if (failed(loadResult)) {
    return failure();
  }
  guard.moduleWasLoaded();

  auto importedMod = guard.getModule();
  if (failed(importedMod)) {
    return failure(); // getModule() already generates an error message
  }

  // Check properties of the included file to ensure symbol resolution will still work.
  auto validationResult = validateLoadedModuleOp(getEmitOpErrFn(this), *importedMod);
  if (failed(validationResult)) {
    return failure();
  }

  return guard.commit();
}

FailureOr<OwningOpRef<ModuleOp>> IncludeOp::openModule() {
  return parseFile(this->getPathAttr(), *this);
}

} // namespace llzk::include
