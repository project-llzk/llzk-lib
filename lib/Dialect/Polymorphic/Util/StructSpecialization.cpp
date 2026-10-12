//===-- StructSpecialization.cpp ------------------------------------------===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2026 Project LLZK
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "llzk/Dialect/Polymorphic/Util/StructSpecialization.h"

#include "llzk/Dialect/Polymorphic/IR/Ops.h"
#include "llzk/Util/SymbolHelper.h"

using namespace mlir;
using namespace llzk::component;

namespace llzk::polymorphic {

FailureOr<StructDefOp> StructSpecializationResolver::resolve(
    SymbolTableCollection &tables, StructType type, int64_t id, Operation *site,
    StringRef diagnosticContext
) {
  if (!definitions) {
    definitions.emplace();
    rootModule.walk([this](StructDefOp definition) {
      auto metadata = definition->getAttrOfType<IntegerAttr>(SPECIALIZATION_ID_ATTR);
      if (metadata && metadata.getValue().isSignedIntN(64) && metadata.getInt() >= 0) {
        (*definitions)[metadata.getInt()].push_back(definition);
      }
    });
  }
  auto target = definitions->find(id);
  if (target == definitions->end()) {
    return site->emitError() << "unknown struct specialization in " << diagnosticContext
                             << " family";
  }
  if (target->second.size() != 1) {
    return site->emitError() << "ambiguous struct specialization in " << diagnosticContext
                             << " family";
  }
  auto definition = target->second.front();
  auto source = type.getDefinition(tables, site);
  if (failed(source)) {
    return failure();
  }
  if (!*source || source->viaInclude()) {
    return site->emitError() << "cannot resolve " << diagnosticContext
                             << " family source; inline includes first";
  }
  auto origin = getPathRelativeToAncestor(source->get(), rootModule);
  if (failed(origin) || definition->getAttr(SPECIALIZATION_ORIGIN_ATTR) != *origin) {
    return site->emitError() << "struct specialization origin does not match " << diagnosticContext
                             << " family";
  }
  return definition;
}

} // namespace llzk::polymorphic
