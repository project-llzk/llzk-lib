//===-- LLZKLayout.cpp ----------------------------------------------------===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2026 Project LLZK
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "llzk/Util/LLZKLayout.h"

#include "llzk/Dialect/Array/IR/Types.h"
#include "llzk/Dialect/Felt/IR/Types.h"
#include "llzk/Dialect/Function/IR/Ops.h"
#include "llzk/Dialect/POD/IR/Types.h"
#include "llzk/Dialect/Polymorphic/IR/Ops.h"
#include "llzk/Dialect/String/IR/Types.h"
#include "llzk/Dialect/Struct/IR/Ops.h"
#include "llzk/Util/SymbolHelper.h"

#include <mlir/IR/Builders.h>
#include <mlir/IR/SymbolTable.h>

#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/DenseSet.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/Support/raw_ostream.h>

using namespace mlir;
using namespace llzk::polymorphic;

namespace {

/// Return whether a storage type contains components whose signals need visiting.
bool containsStruct(Type type) {
  bool found = false;
  type.walk([&found](llzk::component::StructType) { found = true; });
  return found;
}

/// Enumerate declared signal leaves reachable from the main interface.
class LayoutBuilder {
public:
  explicit LayoutBuilder(ModuleOp module) : builder(module.getContext()) {}

  FailureOr<llzk::LLZKLayout> build(ModuleOp module) {
    auto mainType = llzk::getMainInstanceType(module);
    if (failed(mainType)) {
      return failure();
    }
    if (*mainType && mainType->getParams() && !mainType->getParams().empty()) {
      return module.emitError("LLZK layout requires a monomorphized llzk.main");
    }
    auto main = llzk::getMainInstanceDef(tables, module);
    if (failed(main)) {
      return failure();
    }
    if (!*main) {
      return module.emitError("LLZK layout requires a concrete llzk.main");
    }
    if (main->viaInclude()) {
      return module.emitError("inline includes before exporting an LLZK layout");
    }

    auto root = llzk::getTopRootModule(module);
    if (failed(root)) {
      return failure();
    }
    topRoot = *root;

    auto constrain = main->get().getConstrainFuncOp();
    if (!constrain) {
      return main->get().emitError("LLZK layout requires a main constrain function");
    }

    SmallVector<Attribute> path;
    // Arguments to main's constrain come before the member defs in the layout.
    layout.argumentNames.resize(constrain.getNumArguments());
    for (unsigned index = 1; index < constrain.getNumArguments(); ++index) {
      if (auto name = constrain.getArgNameAttr(index)) {
        layout.argumentNames[index] = *name;
      }
      path.assign({builder.getStringAttr("arg"), builder.getI64IntegerAttr(index)});
      if (failed(visitType(constrain.getArgument(index).getType(), constrain, path, true))) {
        return failure();
      }
    }

    path.assign({builder.getStringAttr("main")});
    if (failed(visitStruct(main->get(), path, true))) {
      return failure();
    }

    return std::move(layout);
  }

private:
  LogicalResult visitStruct(
      llzk::component::StructDefOp definition, SmallVectorImpl<Attribute> &path, bool isMain = false
  ) {
    if (llzk::polymorphic::isInTemplate(definition)) {
      return definition.emitError("LLZK layout requires monomorphized struct types");
    }
    if (!activeStructs.insert(definition).second) {
      return definition.emitError("recursive struct storage has no finite LLZK layout");
    }
    for (auto member : definition.getMemberDefs()) {
      path.push_back(member.getSymNameAttr());
      bool isSignal = member.getSignal() || (isMain && member.hasPublicAttr());
      if (failed(visitType(member.getType(), member, path, isSignal))) {
        return failure();
      }
      path.pop_back();
    }
    activeStructs.erase(definition);
    return success();
  }

  /// Index family entries once per declaration; lookup remains independent of
  /// array size and distinguishes different rolled struct fields in a POD.
  LogicalResult loadFamily(Operation *site) {
    if (families.contains(site)) {
      return success();
    }
    auto &entries = families[site];
    Attribute metadata = site->getAttr(FAMILY_ATTR);
    if (!metadata) {
      return success();
    }
    auto records = dyn_cast<ArrayAttr>(metadata);
    if (!records) {
      return site->emitError("invalid poly.family metadata for LLZK layout");
    }
    for (Attribute record : records) {
      auto dict = dyn_cast<DictionaryAttr>(record);
      auto type = dict ? dict.getAs<TypeAttr>(FAMILY_TYPE_KEY) : TypeAttr();
      auto indices = dict ? dict.getAs<ArrayAttr>(FAMILY_INDICES_KEY) : ArrayAttr();
      auto id = dict ? dict.getAs<IntegerAttr>(FAMILY_SPECIALIZATION_KEY) : IntegerAttr();
      if (!type || !isa<llzk::component::StructType>(type.getValue()) || !indices || !id ||
          !id.getValue().isSignedIntN(64) || id.getInt() < 0 ||
          llvm::any_of(indices, [](Attribute attr) {
        auto index = dyn_cast<IntegerAttr>(attr);
        return !index || !index.getType().isIndex() || index.getInt() < 0;
      })) {
        return site->emitError("invalid poly.family entry for LLZK layout");
      }
      if (!entries.try_emplace(std::make_pair(type.getValue(), indices), id.getInt()).second) {
        return site->emitError("duplicate poly.family entry for LLZK layout");
      }
    }
    return success();
  }

  /// Resolve only family specializations needed by the visited storage. Ignore
  /// malformed IDs on unrelated definitions; duplicate IDs matter only when used.
  FailureOr<llzk::component::StructDefOp>
  resolveSpecialization(llzk::component::StructType type, int64_t id, Operation *site) {
    if (!specializationsIndexed) {
      topRoot.walk([this](llzk::component::StructDefOp definition) {
        auto metadata = definition->getAttrOfType<IntegerAttr>(SPECIALIZATION_ID_ATTR);
        if (metadata && metadata.getValue().isSignedIntN(64) && metadata.getInt() >= 0) {
          specializations[metadata.getInt()].push_back(definition);
        }
      });
      specializationsIndexed = true;
    }
    auto target = specializations.find(id);
    if (target == specializations.end()) {
      return site->emitError("unknown struct specialization in LLZK layout family");
    }
    if (target->second.size() != 1) {
      return site->emitError("ambiguous struct specialization in LLZK layout family");
    }
    auto definition = target->second.front();
    auto source = type.getDefinition(tables, site);
    if (failed(source)) {
      return failure();
    }
    if (!*source || source->viaInclude()) {
      return site->emitError("cannot resolve LLZK layout family source; inline includes first");
    }
    auto origin = llzk::getPathRelativeToAncestor(source->get(), topRoot);
    if (failed(origin) || definition->getAttr(SPECIALIZATION_ORIGIN_ATTR) != *origin) {
      return site->emitError("struct specialization origin does not match LLZK layout family");
    }
    return definition;
  }

  LogicalResult visitType(
      Type type, Operation *site, SmallVectorImpl<Attribute> &path, bool isSignal,
      ArrayRef<int64_t> indices = {}
  ) {
    // Unmarked expressions are not signals, but component storage can contain
    // members with their own signal annotations.
    if (!isSignal && !containsStruct(type)) {
      return success();
    }
    if (isa<llzk::felt::FeltType>(type)) {
      auto signalPath = builder.getArrayAttr(path);
      layout.signalIds.try_emplace(signalPath, layout.signalPaths.size());
      layout.signalPaths.push_back(signalPath);
      return success();
    }
    if (auto array = dyn_cast<llzk::array::ArrayType>(type)) {
      for (int64_t size : array.getShape()) {
        if (size < 0) {
          return site->emitError("LLZK layout requires concrete array dimensions");
        }
      }
      SmallVector<int64_t> elementIndices;
      return visitArray(array, site, path, isSignal, elementIndices, 0);
    }
    if (auto pod = dyn_cast<llzk::pod::PodType>(type)) {
      for (auto record : pod.getRecords()) {
        path.push_back(record.getName());
        if (failed(visitType(record.getType(), site, path, isSignal, indices))) {
          return failure();
        }
        path.pop_back();
      }
      return success();
    }
    if (auto structure = dyn_cast<llzk::component::StructType>(type)) {
      if (auto params = structure.getParams(); params && !params.empty()) {
        SmallVector<Attribute> indexAttrs;
        for (int64_t index : indices) {
          indexAttrs.push_back(builder.getIndexAttr(index));
        }
        if (failed(loadFamily(site))) {
          return failure();
        }
        auto found = families[site].find({structure, builder.getArrayAttr(indexAttrs)});
        if (found == families[site].end()) {
          return site->emitError("missing concrete struct family element for LLZK layout");
        }
        auto definition = resolveSpecialization(structure, found->second, site);
        if (failed(definition)) {
          return failure();
        }
        return visitStruct(*definition, path);
      }
      auto definition = structure.getDefinition(tables, site);
      if (failed(definition)) {
        return failure();
      }
      if (!*definition) {
        return site->emitError() << "cannot resolve struct type " << structure
                                 << " for LLZK layout";
      }
      if (definition->viaInclude()) {
        return site->emitError("inline includes before exporting an LLZK layout");
      }
      if (llzk::polymorphic::isInTemplate(definition->get())) {
        return site->emitError("LLZK layout requires monomorphized struct types");
      }
      return visitStruct(definition->get(), path);
    }
    if (type.isIndex() || isa<IntegerType, llzk::string::StringType>(type)) {
      return success();
    }
    return site->emitError() << "unsupported or unresolved type " << type << " in LLZK layout";
  }

  LogicalResult visitArray(
      llzk::array::ArrayType array, Operation *site, SmallVectorImpl<Attribute> &path,
      bool isSignal, SmallVectorImpl<int64_t> &indices, unsigned dimension
  ) {
    if (dimension == array.getRank()) {
      return visitType(array.getElementType(), site, path, isSignal, indices);
    }
    for (int64_t index = 0; index < array.getShape()[dimension]; ++index) {
      path.push_back(builder.getIndexAttr(index));
      indices.push_back(index);
      if (failed(visitArray(array, site, path, isSignal, indices, dimension + 1))) {
        return failure();
      }
      path.pop_back();
      indices.pop_back();
    }
    return success();
  }

  Builder builder;
  SymbolTableCollection tables;
  ModuleOp topRoot;
  bool specializationsIndexed = false;
  llvm::DenseMap<int64_t, SmallVector<llzk::component::StructDefOp, 1>> specializations;
  llvm::DenseMap<Operation *, llvm::DenseMap<std::pair<Type, ArrayAttr>, int64_t>> families;
  llvm::DenseSet<Operation *> activeStructs;
  llzk::LLZKLayout layout;
};

/// Render a structural path, using optional argument names as display labels.
void printPath(ArrayAttr path, ArrayRef<StringAttr> names, llvm::raw_ostream &output) {
  bool isMain = cast<StringAttr>(path[0]).getValue() == "main";
  if (isMain) {
    output << "main";
  } else {
    auto argument = cast<IntegerAttr>(path[1]).getInt();
    if (static_cast<size_t>(argument) < names.size() && names[argument]) {
      output << "arg[";
      names[argument].print(output);
      output << ']';
    } else {
      output << "arg" << argument;
    }
  }
  for (Attribute segment : path.getValue().drop_front(isMain ? 1 : 2)) {
    output << '[';
    if (auto name = dyn_cast<StringAttr>(segment)) {
      name.print(output);
    } else {
      output << cast<IntegerAttr>(segment).getInt();
    }
    output << ']';
  }
}

} // namespace

FailureOr<llzk::LLZKLayout> llzk::buildLLZKLayout(ModuleOp module) {
  return LayoutBuilder(module).build(module);
}

std::optional<uint64_t> llzk::LLZKLayout::getSignalId(ArrayAttr path) const {
  if (!path) {
    return std::nullopt;
  }
  auto found = signalIds.find(path);
  if (found == signalIds.end()) {
    return std::nullopt;
  }
  return found->second;
}

void llzk::printLLZKLayout(const LLZKLayout &layout, llvm::raw_ostream &output) {
  output << "# LLZK layout map v1\n# signals\n";
  for (auto [id, path] : llvm::enumerate(layout.signalPaths)) {
    output << "signal " << id << '\t';
    printPath(path, layout.argumentNames, output);
    output << '\n';
  }
}
