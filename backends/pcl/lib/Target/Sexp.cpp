//===-- Sexp.cpp ------------------------------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2026 Project LLZK
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "Sexp.h"

namespace pcl {

namespace detail {
void List::print(llvm::raw_ostream &os) const {
  os << paren;
  llvm::interleave(elements, os, [&os](auto &elt) { elt->print(os); }, " ");
  switch (paren) {
  case '(':
    os << ')';
    break;
  case '[':
    os << ']';
    break;
  default:
    llvm_unreachable("only ( and [ are allowed");
  }
}
} // namespace detail

void Sexp::print(llvm::raw_ostream &os) const { elt->print(os); }

Sexp Sexp::withSquareBrackets() {
  elt->setParen('[');
  return *this;
}

Sexp SexpCtx::sexp(llvm::ArrayRef<Sexp> elements) {
  static_assert(sizeof(Sexp) == sizeof(detail::SexpElt *));
  detail::SexpElt **buf = allocator.Allocate<detail::SexpElt *>(elements.size());
  memcpy(
      reinterpret_cast<void *>(buf), elements.data(), elements.size() * sizeof(detail::SexpElt *)
  );
// This is a know GCC issue: https://gcc.gnu.org/bugzilla/show_bug.cgi?id=109224
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmismatched-new-delete"
  return Sexp(new (allocator) detail::List(llvm::ArrayRef(buf, elements.size())));
#pragma GCC diagnostic pop
}
} // namespace pcl

llvm::raw_ostream &operator<<(llvm::raw_ostream &os, const pcl::Sexp &sexp) {
  sexp.print(os);
  return os;
}
