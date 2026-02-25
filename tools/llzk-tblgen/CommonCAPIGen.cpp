//===- CommonCAPIGen.cpp - Common utilities for C API generation ----------===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//
//
// Shared command-line options for all CAPI generators (ops, attrs, types)
//
//===----------------------------------------------------------------------===//

#include "CommonCAPIGen.h"

#include <llvm/ADT/StringMap.h>

#include <clang/Basic/FileManager.h>
#include <clang/Basic/LangOptions.h>
#include <clang/Basic/SourceManager.h>
#include <clang/Lex/Lexer.h>
#include <optional>

using namespace mlir;
using namespace clang;

llvm::cl::OptionCategory
    OpGenCat("Options for -gen-op-capi-header, -gen-op-capi-impl, and -gen-op-capi-tests");

llvm::cl::opt<std::string> DialectName(
    "dialect",
    llvm::cl::desc(
        "The dialect name to use for this group of ops. "
        "Must match across header, implementation, and test generation."
    ),
    llvm::cl::cat(OpGenCat)
);

llvm::cl::opt<std::string> FunctionPrefix(
    "prefix",
    llvm::cl::desc(
        "The prefix to use for generated C API function names. "
        "Default is 'mlir'. Must match across header, implementation, and test generation."
    ),
    llvm::cl::init("mlir"), llvm::cl::cat(OpGenCat)
);

llvm::cl::opt<bool> GenIsA(
    "gen-isa", llvm::cl::desc("Generate IsA checks"), llvm::cl::init(true), llvm::cl::cat(OpGenCat)
);

llvm::cl::opt<bool> GenOpBuild(
    "gen-op-build", llvm::cl::desc("Generate operation build(..) functions"), llvm::cl::init(true),
    llvm::cl::cat(OpGenCat)
);

llvm::cl::opt<bool> GenOpOperandGetters(
    "gen-operand-getters", llvm::cl::desc("Generate operand getters for operations"),
    llvm::cl::init(true), llvm::cl::cat(OpGenCat)
);

llvm::cl::opt<bool> GenOpOperandSetters(
    "gen-operand-setters", llvm::cl::desc("Generate operand setters for operations"),
    llvm::cl::init(true), llvm::cl::cat(OpGenCat)
);

llvm::cl::opt<bool> GenOpAttributeGetters(
    "gen-attribute-getters", llvm::cl::desc("Generate attribute getters for operations"),
    llvm::cl::init(true), llvm::cl::cat(OpGenCat)
);

llvm::cl::opt<bool> GenOpAttributeSetters(
    "gen-attribute-setters", llvm::cl::desc("Generate attribute setters for operations"),
    llvm::cl::init(true), llvm::cl::cat(OpGenCat)
);

llvm::cl::opt<bool> GenOpRegionGetters(
    "gen-region-getters", llvm::cl::desc("Generate region getters for operations"),
    llvm::cl::init(true), llvm::cl::cat(OpGenCat)
);

llvm::cl::opt<bool> GenOpResultGetters(
    "gen-result-getters", llvm::cl::desc("Generate result getters for operations"),
    llvm::cl::init(true), llvm::cl::cat(OpGenCat)
);

llvm::cl::opt<bool> GenTypeOrAttrGet(
    "gen-type-attr-get", llvm::cl::desc("Generate get functions for types and attributes"),
    llvm::cl::init(true), llvm::cl::cat(OpGenCat)
);

llvm::cl::opt<bool> GenTypeOrAttrParamGetters(
    "gen-parameter-getters", llvm::cl::desc("Generate parameter getters for types and attributes"),
    llvm::cl::init(true), llvm::cl::cat(OpGenCat)
);

llvm::cl::opt<bool> GenExtraClassMethods(
    "gen-extra-class-methods",
    llvm::cl::desc("Generate C API wrappers for methods in `extraClassDeclaration`"),
    llvm::cl::init(true), llvm::cl::cat(OpGenCat)
);

//===----------------------------------------------------------------------===//
// ClangLexerContext Implementation
//===----------------------------------------------------------------------===//
struct ClangLexerContext::Impl {
  /// C++ language options for lexer configuration
  LangOptions langOpts;
  /// File manager for handling virtual files
  IntrusiveRefCntPtr<FileManager> fileMgr;
  /// Diagnostic IDs for error reporting
  IntrusiveRefCntPtr<DiagnosticIDs> diagIDs;
  /// Diagnostic options for configuring diagnostics
  IntrusiveRefCntPtr<DiagnosticOptions> diagOpts;
  /// Diagnostics engine for handling errors and warnings
  std::unique_ptr<DiagnosticsEngine> diags;
  /// Source manager for tracking file locations
  std::unique_ptr<SourceManager> sourceMgr;
  /// The actual lexer instance
  std::unique_ptr<Lexer> lexer;

  Impl() : diagIDs(new DiagnosticIDs()), diagOpts(new DiagnosticOptions()) {
    // Enable C++ language features for lexing
    langOpts.CPlusPlus = true;
    langOpts.CPlusPlus11 = true;

    FileSystemOptions fileSystemOpts;
    fileMgr = new FileManager(fileSystemOpts);
    diags = std::make_unique<DiagnosticsEngine>(diagIDs, diagOpts);
    sourceMgr = std::make_unique<SourceManager>(*diags, *fileMgr);
  }
};

ClangLexerContext::ClangLexerContext(StringRef source, StringRef bufferName)
    : impl(std::make_unique<Impl>()) {
  if (source.empty()) {
    llvm::errs() << "Warning: ClangLexerContext created with empty source\n";
    return;
  }

  // Create a memory buffer for the input
  std::unique_ptr<llvm::MemoryBuffer> buffer = llvm::MemoryBuffer::getMemBuffer(source, bufferName);
  if (!buffer) {
    llvm::errs() << "Error: Failed to create memory buffer for ClangLexerContext\n";
    return;
  }

  FileID fileID = impl->sourceMgr->createFileID(std::move(buffer), SrcMgr::C_User);
  llvm::MemoryBufferRef bufferRef = impl->sourceMgr->getBufferOrFake(fileID);

  if (bufferRef.getBufferSize() == 0 && !source.empty()) {
    llvm::errs() << "Error: Failed to get buffer from source manager in ClangLexerContext\n";
    return;
  }

  // Create the lexer
  impl->lexer = std::make_unique<Lexer>(fileID, bufferRef, *impl->sourceMgr, impl->langOpts);
  // Enable comment parsing for extraClassDeclaration method extraction
  impl->lexer->SetCommentRetentionState(true);
  lexer = impl->lexer.get();
}

Lexer &ClangLexerContext::getLexer() const {
  assert(lexer && "Lexer not initialized - check isValid() before calling getLexer()");
  return *lexer;
}

SourceManager &ClangLexerContext::getSourceManager() const {
  assert(
      impl && impl->sourceMgr &&
      "SourceManager not initialized - check isValid() before calling getSourceManager()"
  );
  return *impl->sourceMgr;
}

namespace {

static inline bool isAccessModifier(StringRef tokenText) {
  return tokenText == "private" || tokenText == "public" || tokenText == "protected";
}

/// Collect all tokens first for easier lookahead parsing
/// For very large extraClassDeclaration blocks, this could use streaming,
/// but in practice these blocks are small (typically < 100 tokens)
static inline std::vector<Token> tokenize(const ClangLexerContext &lexerCtx) {
  Lexer &lexer = lexerCtx.getLexer();
  std::vector<Token> tokens;
  tokens.reserve(128); // Reasonable default
  for (Token tok; !lexer.LexFromRawLexer(tok);) {
    if (tok.is(tok::eof)) {
      break;
    }
    tokens.push_back(tok);
  }
  return tokens;
}

/// Extract documentation from comment tokens preceding the function declaration.
static inline std::string getDocumentation(
    size_t returnTypeStart, const std::vector<Token> &tokens, const SourceManager &sourceMgr
) {
  std::string documentation;
  for (size_t j = returnTypeStart; j > 0; --j) {
    Token curr = tokens[j - 1];
    if (curr.is(tok::comment)) {
      StringRef comment(sourceMgr.getCharacterData(curr.getLocation()), curr.getLength());
      comment.consume_front("///");
      comment.consume_front("//");
      if (comment.consume_front("/*")) {
        comment.consume_back("*/");
      }
      comment = comment.trim();

      // Trim whitespace
      if (!comment.empty()) {
        if (!documentation.empty()) {
          documentation = "/// " + comment.str() + "\n" + documentation;
        } else {
          documentation = "/// " + comment.str();
        }
      }
    } else if (!curr.is(tok::unknown)) {
      // Stop looking backwards when we hit a non-comment, non-whitespace token
      // that could be part of another declaration
      if (curr.isOneOf(tok::semi, tok::r_brace, tok::l_brace)) {
        break;
      }
      if (curr.is(tok::raw_identifier) && isAccessModifier(curr.getRawIdentifier())) {
        break;
      }
    }
  }
  return documentation;
}

} // namespace

//===----------------------------------------------------------------------===//
// Method Parsing Implementation
//===----------------------------------------------------------------------===//

/// Access level tracking for C++ class declarations
enum class AccessLevel : std::uint8_t { Public, Private, Protected };

/// Track the current access level (public, private, protected) in a C++ class declaration.
/// Updates the access level when encountering access specifiers like "public:", "private:", etc.
///
/// \param i Current token index (will be incremented if access specifier found)
/// \param tokens Vector of all tokens in the token stream
/// \param currentAccess Current access level (updated if access specifier found)
/// \return true if an access specifier was found and processed, false otherwise
static bool
updateAccessLevel(size_t &i, const std::vector<Token> &tokens, AccessLevel &currentAccess) {
  if (i + 1 < tokens.size() && tokens[i + 1].is(tok::colon)) {
    if (tokens[i].is(tok::raw_identifier)) {
      StringRef name = tokens[i].getRawIdentifier();
      if (name == "private") {
        currentAccess = AccessLevel::Private;
        i++; // extra skip for the colon
        return true;
      } else if (name == "public") {
        currentAccess = AccessLevel::Public;
        i++; // extra skip for the colon
        return true;
      } else if (name == "protected") {
        currentAccess = AccessLevel::Protected;
        i++; // extra skip for the colon
        return true;
      }
    }
  }
  return false;
}

/// Extract and parse the return type of a method declaration.
/// Scans backwards from the method name to find the start of the return type,
/// handling access specifiers, comments, and C++ modifiers.
///
/// \param i Index of the method name token
/// \param tokens Vector of all tokens in the token stream
/// \param sourceMgr Source manager for accessing token text
/// \param returnTypeStart Output parameter for the index where return type starts
/// \return The parsed return type string, or empty string if method should be skipped (e.g.,
/// static)
static std::string extractReturnType(
    size_t i, const std::vector<Token> &tokens, const SourceManager &sourceMgr,
    size_t &returnTypeStart
) {
  std::string returnType;
  returnTypeStart = 0;
  bool isStaticMethod = false;

  // Look backwards for return type start, stopping at declaration boundaries
  for (size_t j = i; j > 0; --j) {
    Token curr = tokens[j - 1];
    // Semicolon or right brace indicates lookback has reached the end of a prior declaration.
    if (curr.isOneOf(tok::semi, tok::r_brace)) {
      returnTypeStart = j;
      break;
    }
    // Check for "static" or access modifiers in the return type lookback (both appear as
    // `raw_identifier` in raw token stream).
    if (curr.is(tok::raw_identifier)) {
      StringRef text = curr.getRawIdentifier();
      if (text == "static") {
        isStaticMethod = true;
        break;
      }
      if (tokens[j].is(tok::colon) && isAccessModifier(text)) {
        // In this case, `returnTypeStart` must be after the colon.
        returnTypeStart = j + 1;
        assert(returnTypeStart < tokens.size());
        break;
      }
    }
  }

  // Skip static methods (for now)
  if (isStaticMethod) {
    return "";
  }

  // Adjust `returnTypeStart` for potential comment tokens. Skip as many
  // sequential comments as needed.
  while (tokens[returnTypeStart].is(tok::comment)) {
    returnTypeStart++;
    assert(returnTypeStart <= i);
  }

  // Build return type from tokens, skipping modifiers and comments.
  returnType.reserve(32); // Reasonable default for most type names
  llvm::raw_string_ostream returnTypeStream(returnType);

  for (size_t j = returnTypeStart; j < i; ++j) {
    // Skip comments - they should be extracted as documentation, not part of the return type
    if (tokens[j].is(tok::comment)) {
      continue;
    }

    StringRef tokenText(sourceMgr.getCharacterData(tokens[j].getLocation()), tokens[j].getLength());

    // Skip access specifiers (e.g., "private", "public", "protected").
    if (tokens[j].is(tok::raw_identifier) && isAccessModifier(tokenText)) {
      // If followed by a colon, skip that too
      if (j + 1 < i && tokens[j + 1].is(tok::colon)) {
        j++; // Skip the colon too
      }
      continue;
    }

    // Skip common implementation keywords that indicate we're in code, not a declaration.
    if (tokens[j].is(tok::raw_identifier) && tokenText == "return") {
      // This indicates we've hit implementation code, stop parsing
      returnType.clear();
      break;
    }

    // Skip modifiers and language keywords that shouldn't be in the return type
    if (tokens[j].is(tok::raw_identifier) && isCppModifierKeyword(tokenText)) {
      continue;
    }

    // Skip standalone colons (from lookback to access specifiers)
    if (tokens[j].is(tok::colon)) {
      // Only skip if it's not part of ::
      if (j == 0 || j + 1 >= i || !tokens[j - 1].is(tok::colon)) {
        continue;
      }
    }

    // Add spacing between tokens (but not around ::)
    if (!returnType.empty() && !returnType.ends_with("::") && tokenText != "::" &&
        !tokenText.starts_with("::")) {
      returnTypeStream << ' ';
    }
    returnTypeStream << tokenText;
  }

  // Trim possible whitespace
  return StringRef(returnType).trim().str();
}

/// Parse method parameters from the token stream between parentheses.
/// Extracts parameter types and names, handling default values and complex type syntax.
///
/// \param i Index of the method name token
/// \param tokens Vector of all tokens in the token stream
/// \param sourceMgr Source manager for accessing token text
/// \param closeParenIdx Output parameter for the index of the closing parenthesis
/// \param hasParameters Output parameter indicating if method has parameters
/// \param parameters Output parameter containing parsed parameters
/// \return true if parsing succeeded, false if closing parenthesis not found
static bool parseMethodParameters(
    size_t i, const std::vector<Token> &tokens, const SourceManager &sourceMgr,
    size_t &closeParenIdx, bool &hasParameters, std::vector<MethodParameter> &parameters
) {
  const size_t tokenCount = tokens.size();
  closeParenIdx = tokenCount;
  hasParameters = false;
  parameters.clear();

  // Initialize parenDepth to 1 to account for the opening '(' at tokens[i+1]
  // Start scanning from i+2 (the first token after the opening paren)
  size_t parenDepth = 1;
  for (size_t j = i + 2; j < tokenCount; ++j) {
    if (tokens[j].is(tok::l_paren)) {
      parenDepth++;
    } else if (tokens[j].is(tok::r_paren)) {
      parenDepth--;
      if (parenDepth == 0) {
        closeParenIdx = j;

        // Parse parameters between '(' and ')'
        // Parameters follow the pattern: type name [, type name ...]
        std::vector<Token> paramTokens;
        for (size_t k = i + 2; k < j; ++k) {
          if (k >= tokenCount) {
            break;
          }
          if (!tokens[k].is(tok::comment)) {
            paramTokens.push_back(tokens[k]);
          }
        }
        const size_t paramTokenCount = paramTokens.size();

        // Check if we have actual parameters (excluding just "void")
        if (paramTokenCount == 1) {
          StringRef paramToken(
              sourceMgr.getCharacterData(paramTokens[0].getLocation()), paramTokens[0].getLength()
          );
          if (paramToken != "void") {
            hasParameters = true;
          }
        } else if (paramTokenCount > 1) {
          hasParameters = true;
        }

        // Parse individual parameters
        if (hasParameters) {
          std::string currentParamType;
          std::string currentParamName;
          bool inDefaultValue = false;

          for (size_t k = 0; k < paramTokenCount; ++k) {
            // Check for end of current parameter
            if (paramTokens[k].is(tok::comma)) {
              // Add the current parameter if valid
              if (!currentParamType.empty() && !currentParamName.empty()) {
                parameters.push_back(MethodParameter(currentParamType, currentParamName));
              }
              currentParamType.clear();
              currentParamName.clear();
              inDefaultValue = false;
              continue;
            }
            // Skip tokens that are part of the default value
            if (inDefaultValue) {
              continue;
            }
            // Check for '=' which indicates start of default value
            if (paramTokens[k].is(tok::equal)) {
              inDefaultValue = true;
              continue;
            }

            StringRef tokenText(
                sourceMgr.getCharacterData(paramTokens[k].getLocation()), paramTokens[k].getLength()
            );

            // Identifier token could be part of the type or the parameter name.
            // Simple heuristic: last identifier before comma, equal, or end is the name
            if (paramTokens[k].is(tok::raw_identifier)) {
              if (k + 1 == paramTokenCount ||
                  (k + 1 < paramTokenCount && paramTokens[k + 1].isOneOf(tok::comma, tok::equal))) {
                currentParamName = tokenText.str();
                continue;
              }
            }

            // Other identifiers and other tokens (keywords, ::, *, &, etc.) are part of type.
            llvm::raw_string_ostream paramTypeStream(currentParamType);
            if (!currentParamType.empty() && tokenText != "*" && tokenText != "&" &&
                tokenText != "::" && !tokenText.starts_with("::") &&
                !StringRef(currentParamType).ends_with("::")) {
              paramTypeStream << ' ';
            }
            paramTypeStream << tokenText;
            currentParamType = paramTypeStream.str();
          }

          // Add the last parameter if valid
          if (!currentParamType.empty() && !currentParamName.empty()) {
            parameters.push_back(MethodParameter(currentParamType, currentParamName));
          }
        }

        return true;
      }
    }
  }

  // Couldn't find closing paren
  return false;
}

/// Check if a method is marked as const and find the end of the method declaration.
/// Scans tokens after the closing parenthesis looking for 'const' keyword and declaration end.
///
/// \param closeParenIdx Index of the closing parenthesis
/// \param tokens Vector of all tokens in the token stream
/// \param endIdx Output parameter for the index of the declaration end (';' or '{')
/// \return true if the method is marked const, false otherwise
static bool
checkConstAndFindEnd(size_t closeParenIdx, const std::vector<Token> &tokens, size_t &endIdx) {
  bool isConst = false;
  endIdx = closeParenIdx + 1;

  while (endIdx < tokens.size()) {
    Token curr = tokens[endIdx];
    if (curr.isOneOf(tok::semi, tok::l_brace)) {
      break;
    }
    if (curr.is(tok::raw_identifier) && curr.getRawIdentifier() == "const") {
      isConst = true;
    }
    endIdx++;
  }

  return isConst;
}

/// Parse method declarations from extraClassDeclaration using Clang's Lexer
///
/// This function parses C++ method declarations to extract method signatures.
/// It identifies methods by looking for the pattern: <return_type> <identifier> '(' [params] ')'
/// [const] ';'
///
/// The parsing process:
/// 1. Tokenizes the input using Clang's lexer
/// 2. Tracks access level (public/private/protected) to filter methods
/// 3. For each potential method declaration:
///    - Extracts the return type
///    - Parses parameters (if any)
///    - Checks for const qualifier
///    - Extracts documentation comments
/// 4. Returns a list of successfully parsed methods, excluding:
///    - Static methods
///    - Private/protected methods
///    - Overloaded methods (duplicate names)
SmallVector<ExtraMethod> parseExtraMethods(StringRef extraDecl) {
  if (extraDecl.empty()) {
    return {};
  }

  // Use ClangLexerContext for simplified setup
  const ClangLexerContext lexerCtx(extraDecl, "extraClassDecl");
  if (!lexerCtx.isValid()) {
    llvm::errs() << "Error: Failed to create lexer context for parseExtraMethods\n";
    return {};
  }

  // Store methods uniqued by name to detect and skip overloads (duplicate method names).
  llvm::StringMap<std::optional<ExtraMethod>> methods;

  // Parse tokens to find method declarations
  const std::vector<Token> tokens = tokenize(lexerCtx);
  const size_t tokenCount = tokens.size();
  const SourceManager &sourceMgr = lexerCtx.getSourceManager();

  // Track current access level to avoid generating C API wrappers for private functions. Code
  // generated by `mlir-tblgen` puts the extra declarations in the public section by default.
  AccessLevel currentAccess = AccessLevel::Public;

  for (size_t i = 0; i < tokenCount; ++i) {
    // Skip comments (they'll be extracted separately)
    if (tokens[i].is(tok::comment)) {
      continue;
    }

    // Check for access specifier changes (e.g., "private:", "public:", "protected:").
    if (updateAccessLevel(i, tokens, currentAccess)) {
      continue;
    }

    // Skip private and protected methods - no need to generate C API wrappers
    if (currentAccess != AccessLevel::Public) {
      continue;
    }

    // Look for pattern: [modifiers] <return_type> <identifier> '(' [params] ')' [const] ';'
    // Look for an identifier followed by '('
    if (i + 1 < tokenCount && tokens[i + 1].is(tok::l_paren) && tokens[i].is(tok::raw_identifier)) {
      StringRef methodName = tokens[i].getRawIdentifier();

      // Skip control flow keywords and other language constructs that use parentheses
      if (isCppLanguageConstruct(methodName)) {
        continue;
      }

      // Extract return type (everything before method name)
      size_t returnTypeStart = 0;
      std::string returnType = extractReturnType(i, tokens, sourceMgr, returnTypeStart);

      // Skip static methods (return type is empty if static)
      if (returnType.empty()) {
        continue;
      }

      // Parse method parameters
      size_t closeParenIdx = tokenCount;
      bool hasParameters = false;
      std::vector<MethodParameter> parameters;
      if (!parseMethodParameters(i, tokens, sourceMgr, closeParenIdx, hasParameters, parameters)) {
        // Couldn't find closing paren, skip this method
        continue;
      }

      // Check for 'const' and find declaration end
      size_t endIdx;
      bool isConst = checkConstAndFindEnd(closeParenIdx, tokens, endIdx);

      // Create method struct
      if (!returnType.empty() && !methodName.empty()) {
        if (methods.contains(methodName)) {
          warnSkipped(methodName, "C API does not support method overloading");
          methods[methodName] = std::nullopt;
        } else {
          ExtraMethod method;
          method.returnType = returnType;
          method.methodName = methodName;
          method.documentation = getDocumentation(returnTypeStart, tokens, sourceMgr);
          method.isConst = isConst;
          method.hasParameters = hasParameters;
          method.parameters = parameters;
          methods[methodName] = std::make_optional(method);
        }
      }

      // Skip to end of this declaration for the next iteration.
      i = endIdx;
    }
  }

  // Return valid methods, skipping overloaded names (nullopt entries).
  return llvm::to_vector(
      llvm::map_range(
          llvm::make_filter_range(methods, [](const auto &p) { return p.second.has_value(); }),
          [](const auto &p) { return p.second.value(); }
      )
  );
}

/// Check if a C++ type matches an MLIR type pattern
bool matchesMLIRClass(StringRef cppType, StringRef typeName) {
  if (cppType == typeName) {
    return true;
  }

  // Check for "::mlir::" or "mlir::" prefix
  StringRef prefix = cppType;
  prefix.consume_front("::");
  if (prefix.consume_front("mlir::")) {
    return prefix == typeName;
  }

  return false;
}

/// Convert C++ type to MLIR C API type
std::optional<std::string> tryCppTypeToCapiType(StringRef cppType) {
  cppType = cppType.trim();

  // Primitive types are unchanged
  if (isPrimitiveType(cppType)) {
    return std::make_optional(cppType.str());
  }

  // APInt type is converted via llzk::fromAPInt()
  if (isAPIntType(cppType)) {
    return std::make_optional("int64_t");
  }

  // Pointer type conversions happen via the `unwrap()` function generated
  // by `DEFINE_C_API_PTR_METHODS()` in `mlir/CAPI/IR.h`
  if (cppType.ends_with(" *") || cppType.ends_with("*")) {
    size_t starPos = cppType.rfind('*');
    if (starPos != StringRef::npos) {
      StringRef baseType = cppType.substr(0, starPos).trim();
      if (matchesMLIRClass(baseType, "AsmState")) {
        return std::make_optional("MlirAsmState");
      }
      if (matchesMLIRClass(baseType, "BytecodeWriterConfig")) {
        return std::make_optional("MlirBytecodeWriterConfig");
      }
      if (matchesMLIRClass(baseType, "MLIRContext")) {
        return std::make_optional("MlirContext");
      }
      if (matchesMLIRClass(baseType, "Dialect")) {
        return std::make_optional("MlirDialect");
      }
      if (matchesMLIRClass(baseType, "DialectRegistry")) {
        return std::make_optional("MlirDialectRegistry");
      }
      if (matchesMLIRClass(baseType, "Operation")) {
        return std::make_optional("MlirOperation");
      }
      if (matchesMLIRClass(baseType, "Block")) {
        return std::make_optional("MlirBlock");
      }
      if (matchesMLIRClass(baseType, "OpOperand")) {
        return std::make_optional("MlirOpOperand");
      }
      if (matchesMLIRClass(baseType, "OpPrintingFlags")) {
        return std::make_optional("MlirOpPrintingFlags");
      }
      if (matchesMLIRClass(baseType, "Region")) {
        return std::make_optional("MlirRegion");
      }
      if (matchesMLIRClass(baseType, "SymbolTable")) {
        return std::make_optional("MlirSymbolTable");
      }
    } else {
      llvm::errs() << "Error: Failed to parse pointer type: " << cppType << '\n';
    }
  }

  // These have `wrap()`/`unwrap()` generated by `DEFINE_C_API_METHODS()` in...
  // ... `mlir/CAPI/IR.h`
  if (matchesMLIRClass(cppType, "Attribute")) {
    return std::make_optional("MlirAttribute");
  }
  if (matchesMLIRClass(cppType, "StringAttr")) {
    return std::make_optional("MlirIdentifier");
  }
  if (matchesMLIRClass(cppType, "Location")) {
    return std::make_optional("MlirLocation");
  }
  if (matchesMLIRClass(cppType, "ModuleOp")) {
    return std::make_optional("MlirModule");
  }
  if (matchesMLIRClass(cppType, "Type")) {
    return std::make_optional("MlirType");
  }
  if (matchesMLIRClass(cppType, "Value")) {
    return std::make_optional("MlirValue");
  }
  // ... `mlir/CAPI/AffineExpr.h`
  if (matchesMLIRClass(cppType, "AffineExpr")) {
    return std::make_optional("MlirAffineExpr");
  }
  // ... `mlir/CAPI/AffineMap.h`
  if (matchesMLIRClass(cppType, "AffineMap")) {
    return std::make_optional("MlirAffineMap");
  }
  // ... `mlir/CAPI/IntegerSet.h`
  if (matchesMLIRClass(cppType, "IntegerSet")) {
    return std::make_optional("MlirIntegerSet");
  }
  // ... `mlir/CAPI/Support.h`
  if (matchesMLIRClass(cppType, "TypeID")) {
    return std::make_optional("MlirTypeID");
  }

  // These have `wrap()`/`unwrap()` manually defined in `mlir/CAPI/Support.h`
  if (matchesMLIRClass(cppType, "StringRef")) {
    return std::make_optional("MlirStringRef");
  }
  if (matchesMLIRClass(cppType, "LogicalResult")) {
    return std::make_optional("MlirLogicalResult");
  }

  // Heuristically map custom dialect classes to their C API equivalents
  if (cppType.ends_with("Type")) {
    return std::make_optional("MlirType");
  }
  if (cppType.ends_with("Attr")) {
    return std::make_optional("MlirAttribute");
  }
  if (cppType.ends_with("Op")) {
    return std::make_optional("MlirOperation");
  }

  // Otherwise, not sure how to convert it
  return std::nullopt;
}

// Map C++ type to corresponding C API type
std::string mapCppTypeToCapiType(StringRef cppType) {
  assert(!isArrayRefType(cppType) && "must check `isArrayRefType()` outside");

  std::optional<std::string> capiTypeOpt = tryCppTypeToCapiType(cppType);
  if (capiTypeOpt.has_value()) {
    return capiTypeOpt.value();
  }

  // Otherwise assume it's a type where the C name is a direct translation from the C++ name.
  return toPascalCase(cppType);
}
