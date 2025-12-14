#pragma once

#include "mlir/Pass/Pass.h"

#include <memory>

namespace llzk::zklean {

/// Emits a textual dump of zk-related dialect operations to help debugging
/// pipelines that mix zkLean, zkBuilder, and zkExpr IR.
std::unique_ptr<mlir::Pass> createPrettyPrintZKLeanPass();

} // namespace llzk::zklean
