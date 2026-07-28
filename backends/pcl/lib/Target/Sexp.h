//===-- Sexp.h --------------------------------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2026 Project LLZK
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#pragma once

#include <llvm/ADT/ArrayRef.h>
#include <llvm/Support/Allocator.h>
#include <llvm/Support/raw_ostream.h>

namespace pcl {
class SexpCtx;

namespace detail {
/// Base class for s-expression elements.
class SexpElt {
public:
  virtual ~SexpElt() = default;
  virtual void print(llvm::raw_ostream &os) const = 0;
  virtual void setParen(char) {}
};

/// An atom is a reference to an object that can be printed.
template <typename T> class Atom : public SexpElt {
  T val;
  explicit Atom(T VAL) : val(VAL) {};
  friend SexpCtx;

public:
  void print(llvm::raw_ostream &os) const override { os << val; }
};

/// A list of other s-expressions.
class List : public SexpElt {
  llvm::ArrayRef<SexpElt *> elements;
  char paren = '(';
  explicit List(llvm::ArrayRef<SexpElt *> E) : elements(E) {}
  friend SexpCtx;

public:
  void print(llvm::raw_ostream &os) const override;
  void setParen(char c) override { paren = c; }
};
} // namespace detail

/// Wrapper to keep the raw `SexpElt *` pointers out of the interface.
class Sexp {
  detail::SexpElt *elt;
  Sexp(detail::SexpElt *E) : elt(E) {}

  friend SexpCtx;

public:
  void print(llvm::raw_ostream &os) const;
  Sexp withSquareBrackets();
};

/// Manages the lifetime of s-expressions.
class SexpCtx {
  llvm::BumpPtrAllocator allocator;

public:
  template <typename T> Sexp atom(T val) {
// This is a known GCC issue: https://gcc.gnu.org/bugzilla/show_bug.cgi?id=109224
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmismatched-new-delete"
    return Sexp(new (allocator) detail::Atom<T>(val));
#pragma GCC diagnostic pop
  }

  Sexp sexp(llvm::ArrayRef<Sexp> elements);
};

/// Builder of s-expressions.
///
/// Accumulates s-expressions and dumps the pending expressions on every call to dump.
class Sexps {
  pcl::SexpCtx ctx;
  llvm::SmallVector<pcl::Sexp> sexps;
  unsigned cursor = 0;

  llvm::ArrayRef<pcl::Sexp> asArray() const { return sexps; }

public:
  /// Creates a new atomic s-expression.
  ///
  /// Doesn't add it to the list of pending expressions.
  template <typename T> Sexp atom(T val) { return ctx.atom(val); }

  /// Creates a new s-expression.
  ///
  /// Doesn't add it to the list of pending expressions.
  Sexp sexp(llvm::ArrayRef<Sexp> elements) { return ctx.sexp(elements); }

  /// Pushes an s-expression into the list of pending expressions.
  void push(Sexp sexp) { sexps.push_back(sexp); }

  /// Creates an s-expression and pushes it to the list.
  void push(llvm::ArrayRef<Sexp> elements) { sexps.push_back(sexp(elements)); }

  void dump(llvm::raw_ostream &os) {
    for (auto sexp : asArray().drop_front(cursor)) {
      sexp.print(os);
      os << '\n';
    }
    cursor = sexps.size();
  }
};

} // namespace pcl

llvm::raw_ostream &operator<<(llvm::raw_ostream &os, const pcl::Sexp &sexp);
