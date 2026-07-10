//===-- TransformationPasses.td ----------------------------*- tablegen -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2026 Project LLZK
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#ifndef PCL_TRANSFORMATION_PASSES_TD
#define PCL_TRANSFORMATION_PASSES_TD

include "llzk/Pass/PassBase.td"

def PCLPrintPass
    : LLZKPass<"pcl-print-pass"> {
  let summary = "prints pcl IR into Picus lisp";
  let description = [{
    Convert a PCL module into lisp form suitable for Picus.
  }];
  let constructor = "pcl::createPrintPass()";
}

#endif // PCL_TRANSFORMATION_PASSES_TD
