//===-- Intervals.h ---------------------------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#pragma once

#include "llzk/Util/Field.h"

#include <mlir/Support/LogicalResult.h>

#include <algorithm>

namespace llzk {

/* UnreducedInterval */

class Interval;

/// @brief An inclusive interval [a, b] where a and b are arbitrary integers
/// not necessarily bound to a given field.
class UnreducedInterval {
public:
  UnreducedInterval(const llvm::DynamicAPInt &x, const llvm::DynamicAPInt &y) : a(x), b(y) {}
  /// @brief This constructor is primarily for convenience for unit tests.
  UnreducedInterval(int64_t x, int64_t y) : a(x), b(y) {}

  /* Operations */

  /// @brief Reduce the interval to an interval in the given field.
  /// @param field
  /// @return
  Interval reduce(const Field &field) const;

  /// @brief Compute and return the intersection of this interval and the given RHS.
  /// @param rhs
  /// @return
  UnreducedInterval intersect(const UnreducedInterval &rhs) const;

  /// @brief Compute and return the union of this interval and the given RHS.
  /// @param rhs
  /// @return
  UnreducedInterval doUnion(const UnreducedInterval &rhs) const;

  /// @brief Return the part of the interval that is guaranteed to be less than
  /// the rhs's max value.
  ///
  /// For example, given *this = [0, 7] and rhs = [3, 5], this function would
  /// return [0, 4], since rhs has a max value of 5. If this interval's lower
  /// bound is greater than or equal to the rhs's upper bound, the returned
  /// interval will be "empty" (an interval where a > b). For example,
  /// if *this = [7, 10] and rhs = [0, 7], then no part of *this is less than rhs.
  UnreducedInterval computeLTPart(const UnreducedInterval &rhs) const;

  /// @brief Return the part of the interval that is less than or equal to the
  /// rhs's upper bound.
  ///
  /// For example, given *this = [0, 7] and rhs = [3, 5], this function would
  /// return [0, 5], since rhs has a max value of 5. If this interval's lower
  /// bound is greater than to the rhs's upper bound, the returned
  /// interval will be "empty" (an interval where a > b). For example, if
  /// *this = [8, 10] and rhs = [0, 7], then no part of *this is less than or equal to rhs.
  UnreducedInterval computeLEPart(const UnreducedInterval &rhs) const;

  /// @brief Return the part of the interval that is greater than the rhs's
  /// lower bound.
  ///
  /// For example, given *this = [0, 7] and rhs = [3, 5], this function would
  /// return [4, 7], since rhs has a minimum value of 3. If this interval's
  /// upper bound is less than or equal to the rhs's lower bound, the returned
  /// interval will be "empty" (an interval where a > b). For example,
  /// if *this = [0, 7] and rhs = [7, 10], then no part of *this is greater than rhs.
  UnreducedInterval computeGTPart(const UnreducedInterval &rhs) const;

  /// @brief Return the part of the interval that is greater than or equal to
  /// the rhs's lower bound.
  ///
  /// For example, given *this = [0, 7] and rhs = [3, 5], this function would
  /// return [3, 7], since rhs has a minimum value of 3. If this interval's
  /// upper bound is less than the rhs's lower bound, the returned
  /// interval will be "empty" (an interval where a > b). For example, if
  /// *this = [0, 6] and rhs = [7, 10], then no part of *this is greater than or equal to rhs.
  UnreducedInterval computeGEPart(const UnreducedInterval &rhs) const;

  UnreducedInterval operator-() const;
  friend UnreducedInterval operator+(const UnreducedInterval &lhs, const UnreducedInterval &rhs);
  friend UnreducedInterval operator-(const UnreducedInterval &lhs, const UnreducedInterval &rhs);
  friend UnreducedInterval operator*(const UnreducedInterval &lhs, const UnreducedInterval &rhs);

  /* Comparisons */

  bool overlaps(const UnreducedInterval &rhs) const;

  friend std::strong_ordering
  operator<=>(const UnreducedInterval &lhs, const UnreducedInterval &rhs);

  friend bool operator==(const UnreducedInterval &lhs, const UnreducedInterval &rhs) {
    return std::is_eq(lhs <=> rhs);
  };

  /* Utility */
  llvm::DynamicAPInt getLHS() const { return a; }
  llvm::DynamicAPInt getRHS() const { return b; }

  /// @brief Compute the width of this interval within a given field `f`.
  /// If `a` > `b`, returns 0. Otherwise, returns `b` - `a` + 1.
  llvm::DynamicAPInt width() const;

  /// @brief Returns true iff width() is zero.
  inline bool isEmpty() const { return width() == 0; }

  bool isNotEmpty() const { return !isEmpty(); }

  void print(llvm::raw_ostream &os) const { os << "Unreduced:[ " << a << ", " << b << " ]"; }

  friend llvm::raw_ostream &operator<<(llvm::raw_ostream &os, const UnreducedInterval &ui) {
    ui.print(os);
    return os;
  }

private:
  llvm::DynamicAPInt a, b;
};

/* Interval */

/// @brief Intervals over a finite field. Based on the Picus implementation.
/// An interval may be:
/// - Empty
/// - Entire (meaning any value across the entire field)
/// - Degenerate (meaning it contains a single value)
/// - Or Type A--F. For these types, refer to the below notes:
///
/// A range [a, b] can be split into 2 categories:
/// - Internal: a <= b
/// - External: a > b -- equivalent to [a, p-1] U [0, b]
///
/// Internal range can be further split into 3 categories:
/// (A) a, b < p/2.                                             E.g., [10, 12]
/// (B) a, b > p/2.       OR: a, b \in {-p/2, 0}.               E.g., [p-4, p-2] === [-4, -2]
/// (C) a < p/2, b > p/2.                                       E.g., [p/2 - 5, p/2 + 5]
///
/// External range can be further split into 3 categories:
/// (D) a, b < p/2.       OR: a \in {-p, -p/2}, b \in {0, p/2}. E.g., [12, 10] === [-p+12, 10]
/// (E) a, b > p/2.       OR: a \in {-p/2, 0} , b \in {p/2, p}. E.g., [p-2, p-4] === [-2, p-4]
/// (F) a > p/2, b < p/2. OR: a \in {-p/2, 0} , b \in {0, p/2}. E.g., [p/2 + 5, p/2 - 5]
/// === [-p/2 + 5, p/2 - 5]
///
/// <------------------------------------------------------------->
///   -p           -p/2            0            p/2             p
///      [  A  ]                      [  A  ]
///                       [  B  ]                     [  B  ]
///             [    C    ]                 [    C    ]
///     F     ]              [       F       ]           [      F
/// <------------------------------------------------------------->
///
///   D      ]  [           D            ]  [           D
///          E            ]  [            E              ]  [   E
///
/// For the sake of simplicity, let's just not care about D and E, which covers at least
/// half of the field, and potentially more.
///
/// Now, there are 4 choose 2 possible non-self interactions:
///
/// A acts on B:
/// - intersection: impossible
/// - union: C or F
///
/// A acts on C:
/// - intersection: A
/// - union: C
///
/// A acts on F:
/// - intersection: A
/// - union: F
///
/// B acts on C
/// - intersection: B
/// - union: C
///
/// B acts on F:
/// - intersection: B
/// - union: F
///
/// C acts on F:
/// - intersection: A, B, C, F
///
///   E.g. [p/2 - 10, p/2 + 10] intersects [-p/2 + 2, p/2 - 2]
///
///   = ((-p/2 - 10, -p/2 + 10) intersects (-p/2 + 2, p/2 - 2)) union
///     (( p/2 - 10,  p/2 + 10) intersects (-p/2 + 2, p/2 - 2))
///
///   = (-p/2 + 2, -p/2 + 10) union (p/2 - 10, p/2 - 2)
///
/// - union: don't care for now, we can revisit this later.
class Interval {
public:
  enum class Type : std::uint8_t { TypeA = 0, TypeB, TypeC, TypeF, Empty, Degenerate, Entire };
  static constexpr std::array<std::string_view, 7> TypeNames = {"TypeA", "TypeB", "TypeC",
                                                                "TypeF", "Empty", "Degenerate",
                                                                "Entire"};

  static std::string_view TypeName(Type t) { return TypeNames.at(static_cast<size_t>(t)); }

  /* Static constructors for convenience */

  static Interval Empty(const Field &f) { return Interval(Type::Empty, f); }

  static Interval Degenerate(const Field &f, const llvm::DynamicAPInt &val) {
    return Interval(Type::Degenerate, f, val, val);
  }

  static Interval False(const Field &f) { return Interval::Degenerate(f, f.zero()); }

  static Interval True(const Field &f) { return Interval::Degenerate(f, f.one()); }

  static Interval Boolean(const Field &f) { return Interval::TypeA(f, f.zero(), f.one()); }

  static Interval Entire(const Field &f) { return Interval(Type::Entire, f); }

  static Interval TypeA(const Field &f, const llvm::DynamicAPInt &a, const llvm::DynamicAPInt &b) {
    return Interval(Type::TypeA, f, a, b);
  }

  static Interval TypeB(const Field &f, const llvm::DynamicAPInt &a, const llvm::DynamicAPInt &b) {
    return Interval(Type::TypeB, f, a, b);
  }

  static Interval TypeC(const Field &f, const llvm::DynamicAPInt &a, const llvm::DynamicAPInt &b) {
    return Interval(Type::TypeC, f, a, b);
  }

  static Interval TypeF(const Field &f, const llvm::DynamicAPInt &a, const llvm::DynamicAPInt &b) {
    return Interval(Type::TypeF, f, a, b);
  }

  /// To satisfy the dataflow::ScalarLatticeValue requirements, this class must
  /// be default initializable. The default interval is the full range of values.
  Interval() : Interval(Type::Entire, Field::getField("bn128")) {}

  /// @brief Convert to an UnreducedInterval.
  UnreducedInterval toUnreduced() const;

  /// @brief Get the first side of the interval for TypeF intervals, otherwise
  /// just get the full interval as an UnreducedInterval (with toUnreduced).
  UnreducedInterval firstUnreduced() const;

  /// @brief Get the second side of the interval for TypeA, TypeB, and TypeC intervals.
  /// Using this function is an error for all other interval types.
  UnreducedInterval secondUnreduced() const;

  template <std::pair<Type, Type>... Pairs>
  static bool areOneOf(const Interval &a, const Interval &b) {
    return ((a.ty == std::get<0>(Pairs) && b.ty == std::get<1>(Pairs)) || ...);
  }

  /// Union
  Interval join(const Interval &rhs) const;

  /// Intersect
  Interval intersect(const Interval &rhs) const;

  /// @brief Computes and returns `this` - (`this` & `other`) if the operation
  /// produces a single interval.
  ///
  /// Note that this is an interval difference, not a subtraction operation
  /// like the `operator-` below.
  ///
  /// For example, given `*this` = [1, 10] and `other` = [5, 11], this function
  /// would return [1, 4], as `this` & `other` (the intersection) = [5, 10], so
  /// [1, 10] - [5, 10] = [1, 4].
  ///
  /// For example, given `*this` = [1, 10] and `other` = [5, 6], this function
  /// should return [1, 4] and [7, 10], but we don't support having multiple
  /// disjoint intervals, so `this` is returned as-is.
  Interval difference(const Interval &other) const;

  /* arithmetic ops */

  Interval operator-() const;
  Interval operator~() const;
  friend Interval operator+(const Interval &lhs, const Interval &rhs);
  friend Interval operator-(const Interval &lhs, const Interval &rhs);
  friend Interval operator*(const Interval &lhs, const Interval &rhs);
  friend Interval operator%(const Interval &lhs, const Interval &rhs);
  /// @brief Returns failure if a division-by-zero is encountered.
  friend mlir::FailureOr<Interval> operator/(const Interval &lhs, const Interval &rhs);
  friend Interval operator&(const Interval &lhs, const Interval &rhs);
  friend Interval operator<<(const Interval &lhs, const Interval &rhs);
  friend Interval operator>>(const Interval &lhs, const Interval &rhs);

  /* boolean ops */
  friend Interval boolAnd(const Interval &lhs, const Interval &rhs);
  friend Interval boolOr(const Interval &lhs, const Interval &rhs);
  friend Interval boolXor(const Interval &lhs, const Interval &rhs);
  friend Interval boolNot(const Interval &iv);

  /* Checks and Comparisons */

  inline bool isEmpty() const { return ty == Type::Empty; }
  inline bool isNotEmpty() const { return !isEmpty(); }
  inline bool isDegenerate() const { return ty == Type::Degenerate; }
  inline bool isEntire() const { return ty == Type::Entire; }
  inline bool isTypeA() const { return ty == Type::TypeA; }
  inline bool isTypeB() const { return ty == Type::TypeB; }
  inline bool isTypeC() const { return ty == Type::TypeC; }
  inline bool isTypeF() const { return ty == Type::TypeF; }

  inline bool isBoolFalse() const { return *this == Interval::False(field.get()); }
  inline bool isBoolTrue() const { return *this == Interval::True(field.get()); }
  inline bool isBoolEither() const { return *this == Interval::Boolean(field.get()); }
  inline bool isBoolean() const { return isBoolFalse() || isBoolTrue() || isBoolEither(); }

  template <Type... Types> bool is() const { return ((ty == Types) || ...); }

  bool operator==(const Interval &rhs) const { return ty == rhs.ty && a == rhs.a && b == rhs.b; }

  /* Getters */

  const Field &getField() const { return field.get(); }

  llvm::DynamicAPInt width() const;

  llvm::DynamicAPInt lhs() const { return a; }
  llvm::DynamicAPInt rhs() const { return b; }

  /* Utility */
  struct Hash {
    unsigned operator()(const Interval &i) const {
      return std::hash<const Field *> {}(&i.field.get()) ^ std::hash<Type> {}(i.ty) ^
             llvm::hash_value(i.a) ^ llvm::hash_value(i.b);
    }
  };

  void print(llvm::raw_ostream &os) const;

  friend llvm::raw_ostream &operator<<(llvm::raw_ostream &os, const Interval &i) {
    i.print(os);
    return os;
  }

private:
  Interval(Type t, const Field &f) : field(f), ty(t), a(f.zero()), b(f.zero()) {}
  Interval(Type t, const Field &f, const llvm::DynamicAPInt &lhs, const llvm::DynamicAPInt &rhs)
      : field(f), ty(t), a(f.reduce(lhs)), b(f.reduce(rhs)) {}

  std::reference_wrapper<const Field> field;
  Type ty;
  llvm::DynamicAPInt a, b;
};

} // namespace llzk
