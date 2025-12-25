#!/bin/bash

# 设置编译环境变量
# RUN_RISC_V=0 表示编译为本地 x86-64 架构
# RUN_RISC_V=1 表示编译为 RISC-V 架构

export RUN_RISC_V=0  # 设置为 0 以编译本地 x86-64 代码
export USE_SIM_MODE=0

echo "Environment set for native x86-64 compilation"
echo "RUN_RISC_V=${RUN_RISC_V}"
echo "USE_SIM_MODE=${USE_SIM_MODE}"

# 运行 autotuning
./autotuning.sh "$@"
