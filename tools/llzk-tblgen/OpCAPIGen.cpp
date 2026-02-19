//===- OpCAPIGen.cpp - C API generator for operations ---------------------===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//
//
// OpCAPIGen uses the description of operations to generate C API for the ops.
//
//===----------------------------------------------------------------------===//

#include <mlir/TableGen/GenInfo.h>
#include <mlir/TableGen/Operator.h>

#include <llvm/Support/CommandLine.h>
#include <llvm/Support/FormatVariadic.h>
#include <llvm/TableGen/Error.h>
#include <llvm/TableGen/Record.h>
#include <llvm/TableGen/TableGenBackend.h>

#include <string>
#include <vector>

#include "CommonCAPIGen.h"
#include "OpCAPIParamHelper.h"

using namespace mlir;
using namespace mlir::tblgen;

/// @brief Common between header and implementation generators for operations
struct OpGeneratorData {
  void setOperandName(mlir::StringRef name) { this->operandNameCapitalized = toPascalCase(name); }
  void setAttributeName(mlir::StringRef name) { this->attrNameCapitalized = toPascalCase(name); }
  void setResultName(mlir::StringRef name, int resultIndex) {
    this->resultNameCapitalized =
        name.empty() ? llvm::formatv("Result{0}", resultIndex).str() : toPascalCase(name);
  }
  void setRegionName(mlir::StringRef name, unsigned regionIndex) {
    this->regionNameCapitalized =
        name.empty() ? llvm::formatv("Region{0}", regionIndex).str() : toPascalCase(name);
  }

protected:
  std::string operandNameCapitalized;
  std::string attrNameCapitalized;
  std::string resultNameCapitalized;
  std::string regionNameCapitalized;
};

/// @brief Generator for operation C header files
struct OpHeaderGenerator : public HeaderGenerator, OpGeneratorData {
  using HeaderGenerator::HeaderGenerator;
  virtual ~OpHeaderGenerator() = default;

  void genOpBuildDecl(const std::string &params) const {
    static constexpr char fmt[] = R"(
/* Build a {4}::{2} Operation. */
MLIR_CAPI_EXPORTED MlirOperation {0}{1}_{2}Build(MlirOpBuilder builder, MlirLocation location{3});
)";
    assert(!className.empty() && "className must be set");
    os << llvm::formatv(
        fmt,
        FunctionPrefix,            // {0}
        dialectNameCapitalized,    // {1}
        className,                 // {2}
        params,                    // {3}
        dialect->getCppNamespace() // {4}
    );
  }

  void genOperandGetterDecl() const {
    static constexpr char fmt[] = R"(
/* Get {3} operand from {4}::{2} Operation. */
MLIR_CAPI_EXPORTED MlirValue {0}{1}_{2}Get{3}(MlirOperation op);
)";
    assert(!className.empty() && "className must be set");
    assert(!operandNameCapitalized.empty() && "operandName must be set");
    os << llvm::formatv(
        fmt,
        FunctionPrefix,            // {0}
        dialectNameCapitalized,    // {1}
        className,                 // {2}
        operandNameCapitalized,    // {3}
        dialect->getCppNamespace() // {4}
    );
  }

  void genOperandSetterDecl() const {
    static constexpr char fmt[] = R"(
/* Set {3} operand of {4}::{2} Operation. */
MLIR_CAPI_EXPORTED void {0}{1}_{2}Set{3}(MlirOperation op, MlirValue value);
)";
    assert(!className.empty() && "className must be set");
    assert(!operandNameCapitalized.empty() && "operandName must be set");
    os << llvm::formatv(
        fmt,
        FunctionPrefix,            // {0}
        dialectNameCapitalized,    // {1}
        className,                 // {2}
        operandNameCapitalized,    // {3}
        dialect->getCppNamespace() // {4}
    );
  }

  void genVariadicOperandGetterDecl() const {
    static constexpr char fmt[] = R"(
/* Get number of {3} operands in {4}::{2} Operation. */
MLIR_CAPI_EXPORTED intptr_t {0}{1}_{2}Get{3}Count(MlirOperation op);

/* Get {3} operand at index from {4}::{2} Operation. */
MLIR_CAPI_EXPORTED MlirValue {0}{1}_{2}Get{3}At(MlirOperation op, intptr_t index);
)";
    assert(!className.empty() && "className must be set");
    assert(!operandNameCapitalized.empty() && "operandName must be set");
    os << llvm::formatv(
        fmt,
        FunctionPrefix,            // {0}
        dialectNameCapitalized,    // {1}
        className,                 // {2}
        operandNameCapitalized,    // {3}
        dialect->getCppNamespace() // {4}

    );
  }

  void genVariadicOperandSetterDecl() const {
    static constexpr char fmt[] = R"(
/* Set {3} operands of {4}::{2} Operation. */
MLIR_CAPI_EXPORTED void {0}{1}_{2}Set{3}(MlirOperation op, intptr_t count, MlirValue const *values);
)";
    assert(!className.empty() && "className must be set");
    assert(!operandNameCapitalized.empty() && "operandName must be set");
    os << llvm::formatv(
        fmt,
        FunctionPrefix,            // {0}
        dialectNameCapitalized,    // {1}
        className,                 // {2}
        operandNameCapitalized,    // {3}
        dialect->getCppNamespace() // {4}

    );
  }

  void genAttributeGetterDecl() const {
    static constexpr char fmt[] = R"(
/* Get {3} attribute from {4}::{2} Operation. */
MLIR_CAPI_EXPORTED MlirAttribute {0}{1}_{2}Get{3}(MlirOperation op);
)";
    assert(!className.empty() && "className must be set");
    assert(!attrNameCapitalized.empty() && "attrName must be set");
    os << llvm::formatv(
        fmt,
        FunctionPrefix,            // {0}
        dialectNameCapitalized,    // {1}
        className,                 // {2}
        attrNameCapitalized,       // {3}
        dialect->getCppNamespace() // {4}
    );
  }

  void genAttributeSetterDecl() const {
    static constexpr char fmt[] = R"(
/* Set {3} attribute of {4}::{2} Operation. */
MLIR_CAPI_EXPORTED void {0}{1}_{2}Set{3}(MlirOperation op, MlirAttribute attr);
)";
    assert(!className.empty() && "className must be set");
    assert(!attrNameCapitalized.empty() && "attrName must be set");
    os << llvm::formatv(
        fmt,
        FunctionPrefix,            // {0}
        dialectNameCapitalized,    // {1}
        className,                 // {2}
        attrNameCapitalized,       // {3}
        dialect->getCppNamespace() // {4}
    );
  }

  void genResultGetterDecl() const {
    static constexpr char fmt[] = R"(
/* Get {3} result from {4}::{2} Operation. */
MLIR_CAPI_EXPORTED MlirValue {0}{1}_{2}Get{3}(MlirOperation op);
)";
    assert(!className.empty() && "className must be set");
    assert(!resultNameCapitalized.empty() && "resultName must be set");
    os << llvm::formatv(
        fmt,
        FunctionPrefix,            // {0}
        dialectNameCapitalized,    // {1}
        className,                 // {2}
        resultNameCapitalized,     // {3}
        dialect->getCppNamespace() // {4}
    );
  }

  void genVariadicResultGetterDecl() const {
    static constexpr char fmt[] = R"(
/* Get number of {3} results in {4}::{2} Operation. */
MLIR_CAPI_EXPORTED intptr_t {0}{1}_{2}Get{3}Count(MlirOperation op);

/* Get {3} result at index from {4}::{2} Operation. */
MLIR_CAPI_EXPORTED MlirValue {0}{1}_{2}Get{3}At(MlirOperation op, intptr_t index);
)";
    assert(!className.empty() && "className must be set");
    assert(!resultNameCapitalized.empty() && "resultName must be set");
    os << llvm::formatv(
        fmt,
        FunctionPrefix,            // {0}
        dialectNameCapitalized,    // {1}
        className,                 // {2}
        resultNameCapitalized,     // {3}
        dialect->getCppNamespace() // {4}
    );
  }

  void genRegionGetterDecl() const {
    static constexpr char fmt[] = R"(
/* Get {3} region from {4}::{2} Operation. */
MLIR_CAPI_EXPORTED MlirRegion {0}{1}_{2}Get{3}(MlirOperation op);
)";
    assert(!className.empty() && "className must be set");
    assert(!regionNameCapitalized.empty() && "regionName must be set");
    os << llvm::formatv(
        fmt,
        FunctionPrefix,            // {0}
        dialectNameCapitalized,    // {1}
        className,                 // {2}
        regionNameCapitalized,     // {3}
        dialect->getCppNamespace() // {4}
    );
  }

  void genVariadicRegionGetterDecl() const {
    static constexpr char fmt[] = R"(
/* Get number of {3} regions in {4}::{2} Operation. */
MLIR_CAPI_EXPORTED intptr_t {0}{1}_{2}Get{3}Count(MlirOperation op);

/* Get {3} region at index from {4}::{2} Operation. */
MLIR_CAPI_EXPORTED MlirRegion {0}{1}_{2}Get{3}At(MlirOperation op, intptr_t index);
)";
    assert(!className.empty() && "className must be set");
    assert(!regionNameCapitalized.empty() && "regionName must be set");
    os << llvm::formatv(
        fmt,
        FunctionPrefix,            // {0}
        dialectNameCapitalized,    // {1}
        className,                 // {2}
        regionNameCapitalized,     // {3}
        dialect->getCppNamespace() // {4}
    );
  }
};

/// Generate C API parameter list from operation arguments
///
/// This function builds a comma-separated parameter list for the operation "Build" function.
/// It includes operands, attributes, and result types (if not inferred).
/// Variadic parameters are represented as (count, array) pairs.
static std::string generateCAPIBuildParams(const Operator &op) {
  struct : GenStringFromOpPieces {
    void genResult(
        llvm::raw_ostream &os, const NamedTypeConstraint &result, const std::string &resultName
    ) override {
      if (result.isVariadic()) {
        os << llvm::formatv(", intptr_t {0}Size, MlirType const *{0}Types", resultName);
      } else {
        os << llvm::formatv(", MlirType {0}Type", resultName);
      }
    }
    void genOperand(llvm::raw_ostream &os, const NamedTypeConstraint &operand) override {
      if (operand.isVariadic()) {
        os << llvm::formatv(", intptr_t {0}Size, MlirValue const *{0}", operand.name);
      } else {
        os << llvm::formatv(", MlirValue {0}", operand.name);
      }
    }
    void genAttribute(llvm::raw_ostream &os, const NamedAttribute &attr) override {
      std::optional<std::string> attrType = tryCppTypeToCapiType(attr.attr.getStorageType());
      os << llvm::formatv(", {0} {1}", attrType.value_or("MlirAttribute"), attr.name);
    }
    void genRegion(llvm::raw_ostream &os, const mlir::tblgen::NamedRegion &region) override {
      if (region.isVariadic()) {
        os << llvm::formatv(", unsigned {0}Count", region.name);
      }
    }
  } paramStringGenerator;
  return paramStringGenerator.gen(op);
}

/// Emit C API header
static bool emitOpCAPIHeader(const llvm::RecordKeeper &records, raw_ostream &os) {
  emitSourceFileHeader("Op C API Declarations", os, records);

  OpHeaderGenerator generator("Operation", os);
  generator.genPrologue();

  for (const auto *def : records.getAllDerivedDefinitions("Op")) {
    const Operator op(def);
    const Dialect &dialect = op.getDialect();

    // Generate for the selected dialect only (specified via -dialect command-line option)
    if (dialect.getName() != DialectName) {
      continue;
    }

    generator.setDialectAndClassName(&dialect, op.getCppClassName());

    // Generate "Build" function
    if (GenOpBuild && !op.skipDefaultBuilders()) {
      generator.genOpBuildDecl(generateCAPIBuildParams(op));
    }

    // Generate IsA check
    if (GenIsA) {
      generator.genIsADecl();
    }

    // Generate operand getters and setters
    for (int i = 0, e = op.getNumOperands(); i < e; ++i) {
      const auto &operand = op.getOperand(i);
      generator.setOperandName(operand.name);
      if (operand.isVariadic()) {
        if (GenOpOperandGetters) {
          generator.genVariadicOperandGetterDecl();
        }
        if (GenOpOperandSetters) {
          generator.genVariadicOperandSetterDecl();
        }
      } else {
        if (GenOpOperandGetters) {
          generator.genOperandGetterDecl();
        }
        if (GenOpOperandSetters) {
          generator.genOperandSetterDecl();
        }
      }
    }

    // Generate attribute getters and setters
    for (const auto &namedAttr : op.getAttributes()) {
      generator.setAttributeName(namedAttr.name);
      if (GenOpAttributeGetters) {
        generator.genAttributeGetterDecl();
      }
      if (GenOpAttributeSetters) {
        generator.genAttributeSetterDecl();
      }
    }

    // Generate result getters
    if (GenOpResultGetters) {
      for (int i = 0, e = op.getNumResults(); i < e; ++i) {
        const auto &result = op.getResult(i);
        generator.setResultName(result.name, i);
        if (result.isVariadic()) {
          generator.genVariadicResultGetterDecl();
        } else {
          generator.genResultGetterDecl();
        }
      }
    }

    // Generate region getters
    if (GenOpRegionGetters) {
      for (unsigned i = 0, e = op.getNumRegions(); i < e; ++i) {
        const auto &region = op.getRegion(i);
        generator.setRegionName(region.name, i);
        if (region.isVariadic()) {
          generator.genVariadicRegionGetterDecl();
        } else {
          generator.genRegionGetterDecl();
        }
      }
    }

    // Generate extra class method wrappers
    if (GenExtraClassMethods) {
      generator.genExtraMethods(op.getExtraClassDeclaration());
    }
  }

  generator.genEpilogue();
  return false;
}

/// @brief Generator for operation C implementation files
struct OpImplementationGenerator : public ImplementationGenerator, OpGeneratorData {
  using ImplementationGenerator::ImplementationGenerator;
  virtual ~OpImplementationGenerator() = default;

  /// @brief Generate operation "Build" function implementation
  /// @param operationName The full operation name (e.g., "dialect.opname")
  /// @param params The parameter list for the "Build" function
  /// @param assignments The code to populate the operation state with operands, attributes, etc.
  void genOpBuildImpl(
      const std::string &operationName, const std::string &params, const std::string &assignments
  ) const {
    static constexpr char fmt[] = R"(
MlirOperation {0}{1}_{2}Build(MlirOpBuilder builder, MlirLocation location{3}) {{
  MlirOperationState state = mlirOperationStateGet(mlirStringRefCreateFromCString("{4}"), location);
{5}
  return mlirOpBuilderInsert(builder, mlirOperationCreate(&state));
}
)";
    assert(!className.empty() && "className must be set");
    os << llvm::formatv(
        fmt,
        FunctionPrefix,         // {0}
        dialectNameCapitalized, // {1}
        className,              // {2}
        params,                 // {3}
        operationName,          // {4}
        assignments             // {5}
    );
  }

  void genOperandGetterImpl(int index) const {
    static constexpr char fmt[] = R"(
MlirValue {0}{1}_{2}Get{3}(MlirOperation op) {{
  return mlirOperationGetOperand(op, {4});
}
)";
    assert(!className.empty() && "className must be set");
    assert(!operandNameCapitalized.empty() && "operandName must be set");
    os << llvm::formatv(
        fmt,
        FunctionPrefix,         // {0}
        dialectNameCapitalized, // {1}
        className,              // {2}
        operandNameCapitalized, // {3}
        index                   // {4}
    );
  }

  void genOperandSetterImpl(int index) const {
    static constexpr char fmt[] = R"(
void {0}{1}_{2}Set{3}(MlirOperation op, MlirValue value) {{
  mlirOperationSetOperand(op, {4}, value);
}
)";
    assert(!className.empty() && "className must be set");
    assert(!operandNameCapitalized.empty() && "operandName must be set");
    os << llvm::formatv(
        fmt,
        FunctionPrefix,         // {0}
        dialectNameCapitalized, // {1}
        className,              // {2}
        operandNameCapitalized, // {3}
        index                   // {4}
    );
  }

  void genVariadicOperandGetterImpl(int startIdx) const {
    static constexpr char fmt[] = R"(
intptr_t {0}{1}_{2}Get{3}Count(MlirOperation op) {{
  intptr_t count = mlirOperationGetNumOperands(op);
  assert(count >= {4} && "operand count less than start index");
  return count - {4};
}

MlirValue {0}{1}_{2}Get{3}At(MlirOperation op, intptr_t index) {{
  return mlirOperationGetOperand(op, {4} + index);
}
)";
    assert(!className.empty() && "className must be set");
    assert(!operandNameCapitalized.empty() && "operandName must be set");
    os << llvm::formatv(
        fmt,
        FunctionPrefix,         // {0}
        dialectNameCapitalized, // {1}
        className,              // {2}
        operandNameCapitalized, // {3}
        startIdx                // {4}
    );
  }

  void genVariadicOperandSetterImpl(int startIdx) const {
    static constexpr char fmt[] = R"(
void {0}{1}_{2}Set{3}(MlirOperation op, intptr_t count, MlirValue const *values) {{
  intptr_t numOperands = mlirOperationGetNumOperands(op);
  intptr_t startIdx = {4};

  // Validate bounds
  if (startIdx < 0 || startIdx > numOperands) {{
    return;
  }
  if (count < 0 || count > (std::numeric_limits<intptr_t>::max() - startIdx)) {{
    return;
  }

  intptr_t oldCount = numOperands - startIdx;
  intptr_t newNumOperands = startIdx + count;

  std::vector<MlirValue> newOperands(newNumOperands);

  // Copy operands before this variadic group
  for (intptr_t i = 0; i < startIdx; ++i) {{
    newOperands[i] = mlirOperationGetOperand(op, i);
  }

  // Copy new variadic operands
  for (intptr_t i = 0; i < count; ++i) {{
    newOperands[startIdx + i] = values[i];
  }

  // Copy operands after this variadic group
  for (intptr_t i = startIdx + oldCount; i < numOperands; ++i) {{
    newOperands[i - oldCount + count] = mlirOperationGetOperand(op, i);
  }

  mlirOperationSetOperands(op, newNumOperands, newOperands.data());
}
)";
    assert(!className.empty() && "className must be set");
    assert(!operandNameCapitalized.empty() && "operandName must be set");
    os << llvm::formatv(
        fmt,
        FunctionPrefix,         // {0}
        dialectNameCapitalized, // {1}
        className,              // {2}
        operandNameCapitalized, // {3}
        startIdx                // {4}
    );
  }

  void genAttributeGetterImpl(mlir::StringRef attrName) const {
    static constexpr char fmt[] = R"(
MlirAttribute {0}{1}_{2}Get{3}(MlirOperation op) {{
  return mlirOperationGetAttributeByName(op, mlirStringRefCreateFromCString("{4}"));
}
)";
    assert(!className.empty() && "className must be set");
    assert(!attrNameCapitalized.empty() && "attrName must be set");
    os << llvm::formatv(
        fmt,
        FunctionPrefix,         // {0}
        dialectNameCapitalized, // {1}
        className,              // {2}
        attrNameCapitalized,    // {3}
        attrName                // {4}
    );
  }

  void genAttributeSetterImpl(mlir::StringRef attrName) const {
    static constexpr char fmt[] = R"(
void {0}{1}_{2}Set{3}(MlirOperation op, MlirAttribute attr) {{
  mlirOperationSetAttributeByName(op, mlirStringRefCreateFromCString("{4}"), attr);
}
)";
    assert(!className.empty() && "className must be set");
    assert(!attrNameCapitalized.empty() && "attrName must be set");
    os << llvm::formatv(
        fmt,
        FunctionPrefix,         // {0}
        dialectNameCapitalized, // {1}
        className,              // {2}
        attrNameCapitalized,    // {3}
        attrName                // {4}
    );
  }

  void genResultGetterImpl(int index) const {
    static constexpr char fmt[] = R"(
MlirValue {0}{1}_{2}Get{3}(MlirOperation op) {{
  return mlirOperationGetResult(op, {4});
}
)";
    assert(!className.empty() && "className must be set");
    assert(!resultNameCapitalized.empty() && "resultName must be set");
    os << llvm::formatv(
        fmt,
        FunctionPrefix,         // {0}
        dialectNameCapitalized, // {1}
        className,              // {2}
        resultNameCapitalized,  // {3}
        index                   // {4}
    );
  }

  void genVariadicResultGetterImpl(int startIdx) const {
    static constexpr char fmt[] = R"(
intptr_t {0}{1}_{2}Get{3}Count(MlirOperation op) {{
  intptr_t count = mlirOperationGetNumResults(op);
  assert(count >= {4} && "result count less than start index");
  return count - {4};
}

MlirValue {0}{1}_{2}Get{3}At(MlirOperation op, intptr_t index) {{
  return mlirOperationGetResult(op, {4} + index);
}
)";
    assert(!className.empty() && "className must be set");
    assert(!resultNameCapitalized.empty() && "resultName must be set");
    os << llvm::formatv(
        fmt,
        FunctionPrefix,         // {0}
        dialectNameCapitalized, // {1}
        className,              // {2}
        resultNameCapitalized,  // {3}
        startIdx                // {4}
    );
  }

  void genRegionGetterImpl(unsigned index) const {
    static constexpr char fmt[] = R"(
MlirRegion {0}{1}_{2}Get{3}(MlirOperation op) {{
  return mlirOperationGetRegion(op, {4});
}
)";
    assert(!className.empty() && "className must be set");
    assert(!regionNameCapitalized.empty() && "regionName must be set");
    os << llvm::formatv(
        fmt,
        FunctionPrefix,         // {0}
        dialectNameCapitalized, // {1}
        className,              // {2}
        regionNameCapitalized,  // {3}
        index                   // {4}
    );
  }

  void genVariadicRegionGetterImpl(unsigned startIdx) const {
    static constexpr char fmt[] = R"(
intptr_t {0}{1}_{2}Get{3}Count(MlirOperation op) {{
  intptr_t count = mlirOperationGetNumRegions(op);
  assert(count >= {4} && "region count less than start index");
  return count - {4};
}

MlirRegion {0}{1}_{2}Get{3}At(MlirOperation op, intptr_t index) {{
  return mlirOperationGetRegion(op, {4} + index);
}
)";
    assert(!className.empty() && "className must be set");
    assert(!regionNameCapitalized.empty() && "regionName must be set");
    os << llvm::formatv(
        fmt,
        FunctionPrefix,         // {0}
        dialectNameCapitalized, // {1}
        className,              // {2}
        regionNameCapitalized,  // {3}
        startIdx                // {4}
    );
  }
};

/// Generate C API parameter assignments for operation creation
///
/// This function generates the C code that populates an MlirOperationState with
/// operands, attributes, and result types. It handles both regular and
/// variadic parameters appropriately.
static std::string generateCAPIAssignments(const Operator &op) {
  // Code generated here can use the following variables:
  //  - MlirOpBuilder builder
  //  - MlirLocation location
  //  - MlirOperationState state
  //  - Operand/Attribute/Result parameters per `generateCAPIBuildParams()`
  struct : GenStringFromOpPieces {
    void genResultInferred(llvm::raw_ostream &os) override {
      os << "  mlirOperationStateEnableResultTypeInference(&state);\n";
    }
    void genResult(
        llvm::raw_ostream &os, const NamedTypeConstraint &result, const std::string &resultName
    ) override {
      if (result.isVariadic()) {
        os << llvm::formatv(
            "  mlirOperationStateAddResults(&state, {0}Size, {0}Types);\n", resultName
        );
      } else {
        os << llvm::formatv("  mlirOperationStateAddResults(&state, 1, &{0}Type);\n", resultName);
      }
    }
    void genOperand(llvm::raw_ostream &os, const NamedTypeConstraint &operand) override {
      if (operand.isVariadic()) {
        os << llvm::formatv(
            "  mlirOperationStateAddOperands(&state, {0}Size, {0});\n", operand.name
        );
      } else {
        os << llvm::formatv("  mlirOperationStateAddOperands(&state, 1, &{0});\n", operand.name);
      }
    }
    void genAttributesPrefix(llvm::raw_ostream &os, const mlir::tblgen::Operator &op) override {
      os << "  MlirContext ctx = mlirOpBuilderGetContext(builder);\n";
      os << "  llvm::SmallVector<MlirNamedAttribute, " << op.getNumAttributes()
         << "> attributes;\n";
    }
    void genAttribute(llvm::raw_ostream &os, const NamedAttribute &attr) override {
      // The second parameter to `mlirNamedAttributeGet()` must be an "MlirAttribute". However, if
      // it ends up as "MlirIdentifier", a reinterpret cast is needed. These C structs have the same
      // layout and the C++ mlir::StringAttr is a subclass of mlir::Attribute so the cast is safe.
      std::optional<std::string> attrType = tryCppTypeToCapiType(attr.attr.getStorageType());
      std::string attrValue;
      if (attrType.has_value() && attrType.value() == "MlirIdentifier") {
        attrValue = "reinterpret_cast<MlirAttribute&>(" + attr.name.str() + ")";
      } else {
        attrValue = attr.name.str();
      }

      os << "  if (!mlirAttributeIsNull(" << attrValue << ")) {\n";
      os << "    attributes.push_back(mlirNamedAttributeGet(mlirIdentifierGet(ctx, "
         << "mlirStringRefCreateFromCString(\"" << attr.name << "\")), " << attrValue << "));\n";
      os << "  }\n";
    }
    void genAttributesSuffix(llvm::raw_ostream &os, const mlir::tblgen::Operator &op) override {
      os << "  mlirOperationStateAddAttributes(&state, attributes.size(), attributes.data());\n";
    }
    void genRegionsPrefix(llvm::raw_ostream &os, const mlir::tblgen::Operator &op) override {
      os << "  llvm::SmallVector<MlirRegion, " << op.getNumRegions() << "> regions;\n";
    }
    void genRegion(llvm::raw_ostream &os, const mlir::tblgen::NamedRegion &region) override {
      if (region.isVariadic()) {
        os << llvm::formatv("  for (unsigned i = 0; i < {0}Count; ++i)\n  ", region.name);
      }
      os << "  regions.push_back(mlirRegionCreate());\n";
    }
    void genRegionsSuffix(llvm::raw_ostream &os, const mlir::tblgen::Operator &op) override {
      os << "  mlirOperationStateAddOwnedRegions(&state, regions.size(), regions.data());\n";
    }
  } paramStringGenerator;
  return paramStringGenerator.gen(op);
}

/// Emit C API implementation
static bool emitOpCAPIImpl(const llvm::RecordKeeper &records, raw_ostream &os) {
  emitSourceFileHeader("Op C API Definitions", os, records);

  OpImplementationGenerator generator("Operation", os);

  // Capitalize dialect name for function names
  std::string dialectNameCapitalized = toPascalCase(DialectName);

  for (const auto *def : records.getAllDerivedDefinitions("Op")) {
    const Operator op(def);
    const Dialect &dialect = op.getDialect();
    generator.setDialectAndClassName(&dialect, op.getCppClassName());

    // Generate "Build" function
    if (GenOpBuild && !op.skipDefaultBuilders()) {
      std::string assignments = generateCAPIAssignments(op);
      generator.genOpBuildImpl(op.getOperationName(), generateCAPIBuildParams(op), assignments);
    }

    // Generate IsA check implementation
    if (GenIsA) {
      generator.genIsAImpl();
    }

    // Generate operand getters and setters
    for (int i = 0, e = op.getNumOperands(); i < e; ++i) {
      const auto &operand = op.getOperand(i);
      generator.setOperandName(operand.name);
      if (operand.isVariadic()) {
        if (GenOpOperandGetters) {
          generator.genVariadicOperandGetterImpl(i);
        }
        if (GenOpOperandSetters) {
          generator.genVariadicOperandSetterImpl(i);
        }
      } else {
        if (GenOpOperandGetters) {
          generator.genOperandGetterImpl(i);
        }
        if (GenOpOperandSetters) {
          generator.genOperandSetterImpl(i);
        }
      }
    }

    // Generate attribute getters and setters
    for (const auto &namedAttr : op.getAttributes()) {
      generator.setAttributeName(namedAttr.name);
      if (GenOpAttributeGetters) {
        generator.genAttributeGetterImpl(namedAttr.name);
      }
      if (GenOpAttributeSetters) {
        generator.genAttributeSetterImpl(namedAttr.name);
      }
    }

    // Generate result getters
    if (GenOpResultGetters) {
      for (int i = 0, e = op.getNumResults(); i < e; ++i) {
        const auto &result = op.getResult(i);
        generator.setResultName(result.name, i);
        if (result.isVariadic()) {
          generator.genVariadicResultGetterImpl(i);
        } else {
          generator.genResultGetterImpl(i);
        }
      }
    }

    // Generate region getters
    if (GenOpRegionGetters) {
      for (unsigned i = 0, e = op.getNumRegions(); i < e; ++i) {
        const auto &region = op.getRegion(i);
        generator.setRegionName(region.name, i);
        if (region.isVariadic()) {
          generator.genVariadicRegionGetterImpl(i);
        } else {
          generator.genRegionGetterImpl(i);
        }
      }
    }

    // Generate extra class method implementations
    if (GenExtraClassMethods) {
      generator.genExtraMethods(op.getExtraClassDeclaration());
    }
  }

  return false;
}

static mlir::GenRegistration
    genOpCAPIHeader("gen-op-capi-header", "Generate operation C API header", &emitOpCAPIHeader);

static mlir::GenRegistration
    genOpCAPIImpl("gen-op-capi-impl", "Generate operation C API implementation", &emitOpCAPIImpl);
