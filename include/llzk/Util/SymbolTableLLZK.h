//===-- SymbolTableLLZK.h ------------------------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Adapted from the LLVM Project's mlir/include/mlir/IR/SymbolTable.h
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <mlir/IR/SymbolTable.h>

#include <llvm/ADT/DenseSet.h>

namespace llzk {

/// Get an iterator range for all of the uses, for any symbol, that are nested
/// within the given operation 'from'. This does not traverse into any nested
/// symbol tables. This function returns std::nullopt if there are any unknown
/// operations that may potentially be symbol tables.
std::optional<mlir::SymbolTable::UseRange> getSymbolUses(mlir::Operation *from);
std::optional<mlir::SymbolTable::UseRange> getSymbolUses(mlir::Region *from);

/// Get all of the uses of the given symbol that are nested within the given
/// operation 'from'. This does not traverse into any nested symbol tables.
/// This function returns std::nullopt if there are any unknown operations
/// that may potentially be symbol tables.
std::optional<mlir::SymbolTable::UseRange>
getSymbolUses(mlir::StringAttr symbol, mlir::Operation *from);
std::optional<mlir::SymbolTable::UseRange>
getSymbolUses(mlir::Operation *symbol, mlir::Operation *from);
std::optional<mlir::SymbolTable::UseRange>
getSymbolUses(mlir::StringAttr symbol, mlir::Region *from);
std::optional<mlir::SymbolTable::UseRange>
getSymbolUses(mlir::Operation *symbol, mlir::Region *from);

/// Add all symbols used within the given Type to the provided set.
void getSymbolsUsedIn(mlir::Type t, llvm::SmallDenseSet<mlir::SymbolRefAttr> &symbolsUsed);

/// Add all symbols used within the given Types to the provided set.
void getSymbolsUsedIn(
    mlir::ArrayRef<mlir::Type> types, llvm::SmallDenseSet<mlir::SymbolRefAttr> &symbolsUsed
);

/// Get all symbols used within the given Type.
llvm::SmallDenseSet<mlir::SymbolRefAttr> getSymbolsUsedIn(mlir::Type t);

/// Get all symbols used within the given Types.
llvm::SmallDenseSet<mlir::SymbolRefAttr> getSymbolsUsedIn(mlir::ArrayRef<mlir::Type> types);

/// Return if the given symbol is known to have no uses that are nested
/// within the given operation 'from'. This does not traverse into any nested
/// symbol tables. This function will also return false if there are any
/// unknown operations that may potentially be symbol tables. This doesn't
/// necessarily mean that there are no uses, we just can't conservatively
/// prove it.
bool symbolKnownUseEmpty(mlir::StringAttr symbol, mlir::Operation *from);
bool symbolKnownUseEmpty(mlir::Operation *symbol, mlir::Operation *from);
bool symbolKnownUseEmpty(mlir::StringAttr symbol, mlir::Region *from);
bool symbolKnownUseEmpty(mlir::Operation *symbol, mlir::Region *from);

/// Returns the name of the given symbol operation, or nullptr if no symbol is present.
mlir::StringAttr getSymbolName(mlir::Operation *symbol);
inline mlir::StringAttr getSymbolName(mlir::SymbolOpInterface symbol) {
  return getSymbolName(symbol.getOperation());
}

} // namespace llzk
