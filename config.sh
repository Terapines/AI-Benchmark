#!/bin/bash
# Configuration file for AI-Benchmark
# Users can override these values by setting environment variables before sourcing this file

# Compiler flags
# FIXME: Softmax kernel need without _zvl256b
ARCH="${ARCH:-rv64gcv}"
ABI="${ABI:-lp64d}"

# Compiler paths
# Need change to your own gcc path
GCC_PATH="${GCC_PATH:-${HOME}/Downloads/riscv64-glibc-ubuntu-24.04-gcc/riscv/bin/riscv64-unknown-linux-gnu-g++}"
# Need change to your own zcc path
ZCC_DIR="${ZCC_DIR:-${HOME}/Terapines/ZCC/4.1.7}"
ZCC_PATH="${ZCC_PATH:-${ZCC_DIR}/bin/zcc}"
AR="${AR:-llvm-ar-18}"
# OBJDUMP="llvm-objdump"

# Include directories
# Need change to your own mlir include directory FIXME: Can change to zcc?
MLIR_INCLUDE_DIR="${MLIR_INCLUDE_DIR:-${HOME}/workspace/llvm-project-for-ztc/mlir/include/mlir}"

# Workspace paths
# Need change to your own ztc path
TRITON_PLUGIN_DIRS="${TRITON_PLUGIN_DIRS:-${HOME}/workspace/ztc/}"

# Build GCC command
GCC="${GCC_PATH} -march=${ARCH} -mabi=${ABI} -O3"

# Build ZCC command
ZCC="${ZCC_PATH} -fno-lto --target=riscv64-unknown-linux-gnu -march=${ARCH} -mabi=${ABI} -O3 -I${MLIR_INCLUDE_DIR}"

# Library paths
# FIXME: Need Delete if we don't need tsingmicro module
TX8_DEPS_ROOT="${TX8_DEPS_ROOT:-/share/rd/合作项目/清微智能/tx8_deps}"

LLVM_SYMBOLIZER_PATH="${LLVM_SYMBOLIZER_PATH:-${ZCC_DIR}/bin/llvm-symbolizer}"

# Remote server configuration
# Need change to your own remote server configuration
REMOTE_HOST="${REMOTE_HOST:-root@192.168.4.96}"
REMOTE_PATH="${REMOTE_PATH:-/root/workspace/crux/}"
REMOTE="${REMOTE:-${REMOTE_HOST}:${REMOTE_PATH}}"

# Export environment variables for Python scripts
export TRITON_PLUGIN_DIRS
# FIXME: Need Delete if we don't need tsingmicro module
export TX8_DEPS_ROOT
export LLVM_SYMBOLIZER_PATH

# compiler.py need these environment variables
# Need change to your own ztc path
export LLVM_BINARY_DIR="${LLVM_BINARY_DIR:-${HOME}/workspace/llvm-project-for-ztc/build/bin}"

export TRITON_OPT_PATH=${TRITON_PLUGIN_DIRS}/python/build/cmake.linux-x86_64-cpython-3.12/bin/triton-opt
