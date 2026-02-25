#pragma once

#include <memory>

#include "mlir/Pass/Pass.h"

namespace zklean {

/// Emits a textual dump of zk-related dialect operations to help debugging
/// pipelines that mix zkLean, zkBuilder, and zkExpr IR.
std::unique_ptr<mlir::Pass> createPrettyPrintZKLeanPass();

#define GEN_PASS_REGISTRATION
#include "zklean/Transforms/ZKLeanPasses.h.inc"

} // namespace zklean
