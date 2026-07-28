#!/usr/bin/env python3

# Modified version of
# https://raw.githubusercontent.com/llvm/llvm-project/main/mlir/examples/standalone/test/lit.cfg.py
# from LLVM, which is licensed under Apache 2.0 with LLVM Exceptions.

# -*- Python -*-

import os
import shlex

import lit.formats

from lit.llvm import llvm_config

# Configuration file for the 'lit' test runner.

# name: The name of this test suite.
config.name = 'LLZK'

config.test_format = lit.formats.ShTest(not llvm_config.use_lit_shell)

# suffixes: A list of file extensions to treat as test files.
config.suffixes = ['.mlir', '.llzk', '.smt2', '.td']
config.suffixes.extend(config.extra_suffixes)

# test_source_root: The root path where tests are located.
config.test_source_root = os.path.dirname(__file__)

# test_exec_root: The root path where tests should be run.
config.test_exec_root = os.path.join(config.llzk_obj_root, 'test')

config.substitutions.append(('%PATH%', config.environment['PATH']))
config.substitutions.append(('%input_dir', config.test_source_root))
tablegen_include_dirs = [
    config.llzk_include_dir,
    *config.mlir_include_dirs,
    *config.llvm_include_dirs,
]
config.substitutions.append((
    '%tablegen_includes',
    ' '.join(f'-I{shlex.quote(path)}' for path in tablegen_include_dirs if path),
))

llvm_config.with_system_environment(
    ['HOME', 'INCLUDE', 'LIB', 'TMP', 'TEMP'])

llvm_config.use_default_substitutions()

# excludes: A list of directories to exclude from the testsuite. The 'Inputs'
# subdirectories contain auxiliary inputs for various tests in their parent
# directories.
config.excludes = ['Inputs', 'Examples', 'CMakeLists.txt', 'README.txt', 'LICENSE.txt']

# test_exec_root: The root path where tests should be run.
config.test_exec_root = os.path.join(config.llzk_obj_root, 'test')
config.llzk_tools_dir = os.path.join(config.llzk_obj_root, 'bin')
config.llzk_tblgen_tools_dir = os.path.join(config.llzk_obj_root, 'tools', 'llzk-tblgen')

# Tweak the PATH to include the tools dir.
llvm_config.with_environment('PATH', config.llzk_tools_dir, append_path=True)
llvm_config.with_environment('PATH', config.llzk_tblgen_tools_dir, append_path=True)
llvm_config.with_environment('PATH', config.llvm_tools_dir, append_path=True)

tool_dirs = [config.llzk_tools_dir, config.llzk_tblgen_tools_dir, config.llvm_tools_dir]
tools = [
    "llzk-opt", "llzk-smt-check", "llzk-tblgen", "llzk-translate", "llzk-witgen", "r1cs-opt"
]

llvm_config.add_tool_substitutions(tools, tool_dirs)

llvm_config.with_environment('PYTHONPATH', [
    os.path.join(config.mlir_obj_dir, 'python_packages', 'llzk'),
], append_path=True)

if config.per_test_coverage:
    config.environment["LLVM_PROFILE_FILE"] = "covdata-%p.profraw"

# Limit testing time in the case of non-converging analyses
config.maxIndividualTestTime = 60
