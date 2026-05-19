#pragma once

#include "WitgenDriver.h"

#include <mlir/IR/BuiltinTypes.h>

#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/Support/Error.h>

#include <cstddef>
#include <random>

namespace llzk::witgen {

/// Seed an RNG for random/default witness value materialization.
std::mt19937_64 makeDefaultValueRng(const WitgenOptions &options);

/// Convert one static dimension to `size_t`, rejecting dynamic or invalid sizes.
llvm::Expected<size_t> checkedShapeDimToSize(int64_t dim, llvm::StringRef context);

/// Add two `size_t` values and reject overflow.
llvm::Expected<size_t> checkedAddSize(size_t lhs, size_t rhs, llvm::StringRef context);

/// Multiply two `size_t` values and reject overflow.
llvm::Expected<size_t> checkedMulSize(size_t lhs, size_t rhs, llvm::StringRef context);

/// Return the static element count for one shape, rejecting dynamic sizes.
llvm::Expected<size_t>
getStaticShapeElementCount(llvm::ArrayRef<int64_t> shape, llvm::StringRef context);

/// Return the static element count for one shaped type.
llvm::Expected<size_t> getStaticElementCount(mlir::ShapedType type, llvm::StringRef context);

} // namespace llzk::witgen
