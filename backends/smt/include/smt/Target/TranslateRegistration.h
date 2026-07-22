//===-- TranslateRegistration.h ---------------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2026 Project LLZK
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#pragma once

namespace llzk::smt {
/// Registers the translation from SMT to SMTLIB.
void registerSmtTranslation();
} // namespace llzk::smt
