#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
MODEL_DIR=/Users/iangneal/workspace/circom-benchmarks/tests/libs/circomlib-ml/models
CIRCOM_OUTPUT_DIR=/Users/iangneal/workspace/circom/output

if [[ "${1:-}" == "--run-inner" ]]; then
  shift

  declare -A llzk_files=(
    [averagePooling2D]="${CIRCOM_OUTPUT_DIR}/AveragePooling2D_test_llzk/AveragePooling2D_test.llzk"
    [averagePooling2D_stride]="${CIRCOM_OUTPUT_DIR}/AveragePooling2D_stride_test_llzk/AveragePooling2D_stride_test.llzk"
    [batchNormalization]="${CIRCOM_OUTPUT_DIR}/BatchNormalization_test_llzk/BatchNormalization_test.llzk"
    [conv1D]="${CIRCOM_OUTPUT_DIR}/Conv1D_test_llzk/Conv1D_test.llzk"
    [conv2D]="${CIRCOM_OUTPUT_DIR}/Conv2D_test_llzk/Conv2D_test.llzk"
    [conv2D_stride]="${CIRCOM_OUTPUT_DIR}/Conv2D_stride_test_llzk/Conv2D_stride_test.llzk"
    [flatten2D]="${CIRCOM_OUTPUT_DIR}/Flatten2D_test_llzk/Flatten2D_test.llzk"
    [globalAveragePooling2D]="${CIRCOM_OUTPUT_DIR}/GlobalAveragePooling2D_test_llzk/GlobalAveragePooling2D_test.llzk"
    [globalMaxPooling2D]="${CIRCOM_OUTPUT_DIR}/GlobalMaxPooling2D_test_llzk/GlobalMaxPooling2D_test.llzk"
    [maxPooling2D]="${CIRCOM_OUTPUT_DIR}/MaxPooling2D_test_llzk/MaxPooling2D_test.llzk"
    [maxPooling2D_stride]="${CIRCOM_OUTPUT_DIR}/MaxPooling2D_stride_test_llzk/MaxPooling2D_stride_test.llzk"
    [mnist]="${CIRCOM_OUTPUT_DIR}/mnist_test_llzk/mnist_test.llzk"
    [mnist_convnet]="${CIRCOM_OUTPUT_DIR}/mnist_convnet_test_llzk/mnist_convnet_test.llzk"
    [mnist_latest]="${CIRCOM_OUTPUT_DIR}/mnist_latest_test_llzk/mnist_latest_test.llzk"
    [mnist_latest_precision]="${CIRCOM_OUTPUT_DIR}/mnist_latest_precision_test_llzk/mnist_latest_precision_test.llzk"
    [model1]="${CIRCOM_OUTPUT_DIR}/model1_test_llzk/model1_test.llzk"
    [sumPooling2D]="${CIRCOM_OUTPUT_DIR}/SumPooling2D_test_llzk/SumPooling2D_test.llzk"
    [sumPooling2D_stride]="${CIRCOM_OUTPUT_DIR}/SumPooling2D_stride_test_llzk/SumPooling2D_stride_test.llzk"
  )

  cases=(
    # averagePooling2D
    # averagePooling2D_stride
    # batchNormalization
    # conv1D
    # conv2D
    # conv2D_stride
    # flatten2D
    # globalAveragePooling2D
    # globalMaxPooling2D
    # maxPooling2D
    # maxPooling2D_stride
    # mnist
    # mnist_convnet
    # mnist_latest
    # mnist_latest_precision
    model1
    sumPooling2D
    sumPooling2D_stride
  )

  failed_cases=()

  for case_name in "${cases[@]}"; do
    input_json="${MODEL_DIR}/${case_name}_input.json"
    output_json="${MODEL_DIR}/${case_name}_output.json"
    llzk_file="${llzk_files[${case_name}]}"

    [[ -f "${input_json}" ]] || { echo "Missing input JSON: ${input_json}" >&2; exit 1; }
    [[ -f "${output_json}" ]] || { echo "Missing output JSON: ${output_json}" >&2; exit 1; }
    [[ -f "${llzk_file}" ]] || { echo "Missing LLZK file: ${llzk_file}" >&2; exit 1; }

    printf '\n==> %s\n' "${case_name}"
    if llzk-witgen --backend=execution-engine \
      "${llzk_file}" \
      --inputs "${input_json}" \
      --check-output "${output_json}"; then
      echo "PASS: ${case_name}"
    else
      echo "FAIL: ${case_name}" >&2
      failed_cases+=("${case_name}")
    fi
  done

  if (( ${#failed_cases[@]} > 0 )); then
    printf '\nFailed cases (%d):\n' "${#failed_cases[@]}" >&2
    printf '  %s\n' "${failed_cases[@]}" >&2
    exit 1
  fi

  echo
  echo "All execution-engine checks passed."
  exit 0
fi

cd "${SCRIPT_DIR}"
nix build -L
nix develop --command bash "${BASH_SOURCE[0]}" --run-inner "$@"
