//===- CAPIGenRegistration.h - C API generator registration -----*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2026 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#pragma once

namespace llzk {

/// Register shared C API options before command-line parsing.
/// Initialization failures propagate to the caller; do not retry after failure.
void registerCAPIOptions();

/// Register the attribute C API generators before command-line parsing.
void registerAttrCAPIGenerators();

/// Register the attribute C API test generator before command-line parsing.
void registerAttrCAPITestGenerator();

/// Register the type C API generators before command-line parsing.
void registerTypeCAPIGenerators();

/// Register the type C API test generator before command-line parsing.
void registerTypeCAPITestGenerator();

/// Register the enum C API generators before command-line parsing.
void registerEnumCAPIGenerators();

/// Register the enum C API test generator before command-line parsing.
void registerEnumCAPITestGenerator();

/// Register the operation C API generators before command-line parsing.
void registerOpCAPIGenerators();

/// Register the operation C API test generator before command-line parsing.
void registerOpCAPITestGenerator();

/// Register the dialect C API test generator before command-line parsing.
void registerDialectCAPITestGenerator();

} // namespace llzk
