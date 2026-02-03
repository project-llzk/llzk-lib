//===-- LLZKLoweringUtils.h -------------------------------------*- C++ -*-===//
//
// Shared utilities for lowering passes in the LLZK project.
//
//===----------------------------------------------------------------------===//

#ifndef LLZK_TRANSFORMS_LOWERING_UTILS_H
#define LLZK_TRANSFORMS_LOWERING_UTILS_H

#include "llzk/Dialect/Constrain/IR/Ops.h"
#include "llzk/Dialect/Felt/IR/Ops.h"
#include "llzk/Dialect/Function/IR/Ops.h"

#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Value.h>
#include <mlir/Support/LogicalResult.h>

#include <llvm/ADT/DenseMap.h>

namespace llzk {

struct AuxAssignment {
  std::string auxMemberName;
  mlir::Value computedValue;
};

mlir::Value rebuildExprInCompute(
    mlir::Value val, function::FuncDefOp computeFunc, mlir::OpBuilder &builder,
    llvm::DenseMap<mlir::Value, mlir::Value> &memo
);

mlir::LogicalResult
checkForAuxMemberConflicts(component::StructDefOp structDef, llvm::StringRef auxPrefix);

component::MemberDefOp addAuxMember(component::StructDefOp structDef, llvm::StringRef name);

unsigned getFeltDegree(mlir::Value val, llvm::DenseMap<mlir::Value, unsigned> &memo);

/// Replaces all *subsequent uses* of `oldVal` with `newVal`, starting *after* `afterOp`.
///
/// Specifically:
/// - Uses of `oldVal` in operations that come **after** `afterOp` in the same block are replaced.
/// - Uses in `afterOp` itself are **not replaced** (to avoid self-trivializing rewrites).
/// - Uses in other blocks are replaced (if applicable).
///
/// Typical use case:
/// - You introduce an auxiliary value (e.g., via EmitEqualityOp) and want to replace
///   all *later* uses of the original value while preserving the constraint itself.
///
/// \param oldVal  The original value whose uses should be redirected.
/// \param newVal  The new value to replace subsequent uses with.
/// \param afterOp The operation after which uses of `oldVal` will be replaced.
void replaceSubsequentUsesWith(mlir::Value oldVal, mlir::Value newVal, mlir::Operation *afterOp);

} // namespace llzk

#endif // LLZK_TRANSFORMS_LOWERING_UTILS_H
