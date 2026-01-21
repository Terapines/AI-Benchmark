#!/bin/bash

DIR=`dirname $0`
# Source configuration file
source ${DIR}/config.sh
SRC_DIR=${DIR}/src
BUILD_DIR=${DIR}/build

# "Benchmark": Various input shape
# "Accuracy": Specified input shape
MODE="Accuracy"
# Update run.sh mode
sed -i "s/MODE=\(\".*\"\)/MODE=\"${MODE}\"/g" ${DIR}/run.sh
sed -i "s/MODE=\(\".*\"\)/MODE=\"${MODE}\"/g" ${DIR}/report.sh

# Python virtual environment for triton kernel compilation
PYC="python"
# TRITON_PLUGIN_DIRS is now from config.sh
TRITON_PYTHON_VENV=${TRITON_PLUGIN_DIRS}/.venv
# triton-cpu kernel launcher
KERNEL_LAUNCHER_INCLUDE_DIR=${BUILD_DIR}/aux/include


### FIXME: Choose which kernels should be compiled
# C_KERNELS=`ls ${SRC_DIR}/c/*.cpp`
# TRITON_KERNELS=`ls ${SRC_DIR}/triton/*.py`
# DRIVERS=`ls ${SRC_DIR}/main/*.cpp`

### FIXME: Choose which kernels should be compiled
# Array of "c_kernel triton_kernel driver_path" entries
# FIXME: Need add more test cases
drivers=(
  #"${SRC_DIR}/c/matmul.cpp ${SRC_DIR}/triton/matmul.py ${SRC_DIR}/main/matmul.cpp"
  #"${SRC_DIR}/c/softmax.cpp ${SRC_DIR}/triton/softmax.py ${SRC_DIR}/main/softmax_kernel.cpp"
  #"${SRC_DIR}/c/correlation.cpp ${SRC_DIR}/triton/correlation.py ${SRC_DIR}/main/correlation.cpp"
  #"${SRC_DIR}/c/dropout.cpp ${SRC_DIR}/triton/dropout.py ${SRC_DIR}/main/dropout.cpp"
  #"${SRC_DIR}/c/layernorm.cpp ${SRC_DIR}/triton/layernorm.py ${SRC_DIR}/main/layernorm.cpp"
  "${SRC_DIR}/c/resize.cpp ${SRC_DIR}/triton/resize.py ${SRC_DIR}/main/resize.cpp"
)

# Default clean build directory
DO_CLEAN="--clean"

# Helper function
help()
{
cat <<END
Build AI-Benchmark.

Usage: ./build.sh [--clean | --no-clean]
                            [--help]

Options:
  --clean | --no-clean
    Should this script clean build dir before building testsuite
    Default: $DO_CLEAN

  --help
    Print this help message and exit
END
}

# build support library
build_support_lib() {
  ${COMPILER} -fPIC -I ${DIR}/include -c ${SRC_DIR}/support/*.cpp -o ${OBJ_DIR}/support.o
  ${AR} rcs ${LIB_DIR}/libsupport.a ${OBJ_DIR}/support.o
}

# build c kernel
build_c_kernel_lib() {
  for kernel in ${C_KERNELS[@]}; do
    name=`basename ${kernel} .cpp`
    echo ${kernel}
    # FIXME: Maybe need find a good way to use -fopenmp
    if [[ "${COMPILER}" == *"zcc"* ]]; then
      # For zcc: split into two steps
      # Step 1: Generate .ll file from .cpp
      ${COMPILER} -fPIC -I ${DIR}/include -emit-llvm -S ${kernel} -fno-unroll-loops -fopenmp=libomp -o ${OBJ_DIR}/${name}.ll
      echo "${COMPILER} -fPIC -I ${DIR}/include -emit-llvm -S ${kernel} -fopenmp=libomp -o ${OBJ_DIR}/${name}.ll"
      ${COMPILER} -fPIC -I ${DIR}/include -S ${kernel} -fno-unroll-loops -fopenmp=libomp -o ${OBJ_DIR}/${name}.s
      echo "${COMPILER} -fPIC -I ${DIR}/include -S ${kernel} -fopenmp=libomp -o ${OBJ_DIR}/${name}.s"
      # Step 2: Generate .o file from .ll
      ${COMPILER} -fPIC -I ${DIR}/include -c ${kernel} -fno-unroll-loops -fopenmp=libomp -o ${OBJ_DIR}/${name}.o
    else
      # For other compilers (e.g., gcc): single step
      ${COMPILER} -fPIC -I ${DIR}/include -S ${kernel} -fopenmp -lgomp -o ${OBJ_DIR}/${name}.s
      ${COMPILER} -fPIC -I ${DIR}/include -c ${kernel} -fopenmp -lgomp -o ${OBJ_DIR}/${name}.o
      echo "${COMPILER} -fPIC -I ${DIR}/include -c ${kernel} -fopenmp -lgomp -o ${OBJ_DIR}/${name}.o"
    fi
  done

  find ${OBJ_DIR} -not -name "support.o" -name "*.o" | xargs ${AR} rcs ${LIB_DIR}/libkernel.a
}

# build triton kernel
build_triton_kernel_lib() {
  source ${TRITON_PYTHON_VENV}/bin/activate

  for kernel in ${TRITON_KERNELS[@]}; do
    name=`basename ${kernel} .py`

    KERNEL_AUX_FILE_DIR=${BUILD_DIR}/aux/src/${name}/
    mkdir -p ${KERNEL_AUX_FILE_DIR}

    echo ${kernel}
    # compile triton kernel: .py --> .llir + launcher.cpp
    # TRITON_ALWAYS_COMPILE=1 MLIR_ENABLE_DUMP=1
    KERNEL_LAUNCHER_INCLUDE_DIR=${KERNEL_LAUNCHER_INCLUDE_DIR} KERNEL_AUX_FILE_DIR=${KERNEL_AUX_FILE_DIR} RUN_RISC_V=${RUN_RISC_V} ${PYC} ${kernel}
    echo "KERNEL_LAUNCHER_INCLUDE_DIR=${KERNEL_LAUNCHER_INCLUDE_DIR} KERNEL_AUX_FILE_DIR=${KERNEL_AUX_FILE_DIR} RUN_RISC_V=${RUN_RISC_V} ${PYC} ${kernel}"

    # TODO: Update Clang version
    # For now, we just replace the trunc n[us]w with trunc
    # Also remove captures(none) attributes for RISC-V compatibility
    sed -i 's/trunc nuw nsw/trunc/g; s/trunc nuw/trunc/g; s/trunc nsw/trunc/g; s/\s*captures(none)//g' ${KERNEL_AUX_FILE_DIR}/*.llir

    # build triton kernel: .llir --> .o
    for kernel_ir in ${KERNEL_AUX_FILE_DIR}/*.llir; do
      kernel_name=`basename ${kernel_ir} .llir`
      echo ${kernel_ir}
      # llc -march=riscv64 -mattr=+d,v  ${kernel_ir} -o ${KERNEL_AUX_FILE_DIR}/${kernel_name}.s
      # z++ -march=rv64gcv -fno-lto --target=riscv64-unknown-linux-gnu -S -x ir  -O2 ${kernel_ir} -mllvm --riscv-disable-rvv-fixedlen=false -mrvv-vector-bits=256 -o ${KERNEL_AUX_FILE_DIR}/${kernel_name}.s

      echo "${COMPILER} -O3 -S -x ir /home/xinyi/workspace/AI-Benchmark/resize_kernel.llir -fopenmp=libomp -mllvm --riscv-disable-rvv-fixedlen=false -mrvv-vector-bits=128 -o ${KERNEL_AUX_FILE_DIR}/${kernel_name}.s"
      ${COMPILER} -O3 -S -x ir ${kernel_ir} -fopenmp=libomp -mllvm --riscv-disable-rvv-fixedlen=false -mrvv-vector-bits=128 -o ${KERNEL_AUX_FILE_DIR}/${kernel_name}.s

      ${COMPILER} -c -o ${OBJ_DIR}/${kernel_name}.o ${KERNEL_AUX_FILE_DIR}/${kernel_name}.s
    done

    # build triton laucher: launcher.cpp --> .o
    for kernel_launcher in ${KERNEL_AUX_FILE_DIR}/*.cpp; do
      launcher_name=`basename ${kernel_launcher} .cpp`
      # FIXME: Maybe need find a good way to use -fopenmp
      ${COMPILER} -I ${DIR}/include -I ${KERNEL_LAUNCHER_INCLUDE_DIR} -c -flto ${kernel_launcher} -fopenmp=libomp -o ${OBJ_DIR}/${launcher_name}.o
    done

  done

  # # 添加：如果存在手写的 softmax_kernel.s 文件，将其编译并添加到库中
  # if [ -f "/home/xinyi/workspace/AI-Benchmark/softmax_kernel.s" ]; then
  #   echo "Found custom softmax_kernel.s, compiling and adding to libkernel.a"
  #   ${COMPILER} -c -o ${OBJ_DIR}/softmax_kernel.o /home/xinyi/workspace/AI-Benchmark/softmax_kernel.s
  # fi

  find ${OBJ_DIR} -not -name "support.o" -name "*.o" | xargs ${AR} rcs ${BUILD_DIR}/lib/triton/libkernel.a
}

create_dir_hierarchy(){
  rm -rf ${LIB_DIR}
  rm -rf ${BIN_DIR}
  rm -rf ${OBJ_DIR}
  mkdir -p ${LIB_DIR}
  mkdir -p ${BIN_DIR}
  mkdir -p ${OBJ_DIR}
}

# build driver
build_driver(){
  case $1 in
    zcc)
      COMPILER=${ZCC}
      LIB_DIR=${BUILD_DIR}/lib/zcc
      BIN_DIR=${BUILD_DIR}/bin/zcc
      OBJ_DIR=${BUILD_DIR}/obj/zcc
      KERNEL_ENABLE=C_KERNEL_ENABLE
      ;;
    gcc)
      COMPILER=${GCC}
      LIB_DIR=${BUILD_DIR}/lib/gcc
      BIN_DIR=${BUILD_DIR}/bin/gcc
      OBJ_DIR=${BUILD_DIR}/obj/gcc
      KERNEL_ENABLE=C_KERNEL_ENABLE
      ;;
    triton)
      COMPILER=${ZCC}
      LIB_DIR=${BUILD_DIR}/lib/triton
      BIN_DIR=${BUILD_DIR}/bin/triton
      OBJ_DIR=${BUILD_DIR}/obj/triton
      KERNEL_ENABLE=TRITON_KERNEL_ENABLE
      ;;
    ?*)
      echo "Unknwon option"
      exit -1
      ;;
  esac

  create_dir_hierarchy
  if [ "${KERNEL_ENABLE}" == "C_KERNEL_ENABLE" ]; then
    build_c_kernel_lib
  else
    mkdir -p ${KERNEL_LAUNCHER_INCLUDE_DIR}
    mkdir -p ${BUILD_DIR}/aux/src
    build_triton_kernel_lib
  fi

  build_support_lib

  # Benchmark mode don't check accurary since io operation is slow
  if [ "${MODE}" == "Accuracy" ]; then
    COMPILER+=" -DCHECK_ACCURACY "
  fi

  for main in ${DRIVERS[@]}; do
    name=`basename ${main} .cpp`
    echo ${main}

    KERNEL_BIN_DIR=${BIN_DIR}/${name}/
    mkdir -p ${KERNEL_BIN_DIR}

    # Compile driver
    # .elf suffix to avoid scp problem(same name dir and kernel)
    # FIXME:lmlir_c_runner_utils is for memrefcopy function in ztc, maybe we need to remove it in the future
    if [[ "${COMPILER}" == *"zcc"* ]]; then
      echo "${COMPILER} ${main} -I ${DIR}/include -I ${KERNEL_LAUNCHER_INCLUDE_DIR} -fno-unroll-loops -fopenmp=libomp -L ${LIB_DIR} -L/share/rd/temp/ztc-mlir-lib -lmlir_c_runner_utils -lmlir_float16_utils -lstdc++ -lm -lkernel -lsupport -latomic -std=c++17 -D${KERNEL_ENABLE} -fPIC -o ${KERNEL_BIN_DIR}/${name}.elf"
      ${COMPILER} /home/xinyi/workspace/AI-Benchmark/resize-kernel.s ${main} -I ${DIR}/include -I ${KERNEL_LAUNCHER_INCLUDE_DIR} -fno-unroll-loops -fopenmp=libomp -L ${LIB_DIR} -L/share/rd/temp/ztc-mlir-lib -lmlir_c_runner_utils -lmlir_float16_utils -lstdc++ -lm -lkernel -lsupport -latomic -std=c++17 -D${KERNEL_ENABLE} -fPIC -o ${KERNEL_BIN_DIR}/${name}.elf
    elif [[ "${COMPILER}" == *"gcc"* ]]; then
      ${COMPILER} ${main} -I ${DIR}/include -I ${KERNEL_LAUNCHER_INCLUDE_DIR} -fopenmp -L ${LIB_DIR} -L/share/rd/temp/ztc-mlir-lib -lmlir_c_runner_utils -lmlir_float16_utils -lstdc++ -lm -lkernel -lgomp -lsupport -latomic -std=c++17 -D${KERNEL_ENABLE} -fPIC -o ${KERNEL_BIN_DIR}/${name}.elf
    else
      echo "wrong compiler"
    fi
    # ${OBJDUMP} -d ${KERNEL_BIN_DIR}/${name}.elf &> ${KERNEL_BIN_DIR}/${name}.elf.s

    # Data shape config
    cp ${SRC_DIR}/main/${name}.cfg  ${KERNEL_BIN_DIR}
  done
}


# build
# ├── aux    // When use MAGIC compiler, we may generate *.o or lib directly.
# │   ├── include
# │   │   ├── _layer_norm_bwd_dwdb_launcher.h
# │   │   ├── _layer_norm_bwd_dx_fused_launcher.h
# │   │   └── _layer_norm_fwd_fused_launcher.h
# │   └── src
# │       └── layernorm
# │           ├── _layer_norm_bwd_dwdb_launcher.cpp
# │           ├── _layer_norm_bwd_dwdb.llir
# │           ├── _layer_norm_bwd_dwdb.s
# │           ├── _layer_norm_bwd_dx_fused_launcher.cpp
# │           ├── _layer_norm_bwd_dx_fused.llir
# │           ├── _layer_norm_bwd_dx_fused.s
# │           ├── _layer_norm_fwd_fused_launcher.cpp
# │           ├── _layer_norm_fwd_fused.llir
# │           └── _layer_norm_fwd_fused.s
# ├── bin
# │   ├── gcc
# │   │   └── layernorm
# │   │       ├── layernorm.cfg
# │   │       └── layernorm.elf
# │   ├── triton
# │   │   └── layernorm
# │   │       ├── layernorm.cfg
# │   │       └── layernorm.elf
# │   └── zcc
# │   │   └── layernorm
# │   │       ├── layernorm.cfg
# │   │       └── layernorm.elf
# ├── lib
# │   ├── gcc
# │   │   ├── libkernel.a
# │   │   └── libsupport.a
# │   ├── triton
# │   │   ├── libkernel.a
# │   │   └── libsupport.a
# │   └── zcc
# │       ├── libkernel.a
# │       └── libsupport.a
# └── obj
#     ├── gcc
#     │   ├── _layer_norm_bwd_dwdb_launcher.o
#     │   ├── _layer_norm_bwd_dwdb.o
#     │   ├── _layer_norm_bwd_dx_fused_launcher.o
#     │   ├── _layer_norm_bwd_dx_fused.o
#     │   ├── _layer_norm_fwd_fused_launcher.o
#     │   ├── _layer_norm_fwd_fused.o
#     │   └── support.o
#     ├── triton
#     │   ├── _layer_norm_bwd_dwdb_launcher.o
#     │   ├── _layer_norm_bwd_dwdb.o
#     │   ├── _layer_norm_bwd_dx_fused_launcher.o
#     │   ├── _layer_norm_bwd_dx_fused.o
#     │   ├── _layer_norm_fwd_fused_launcher.o
#     │   ├── _layer_norm_fwd_fused.o
#     │   └── support.o
#     └── zcc
#         ├── _layer_norm_bwd_dwdb_launcher.o
#         ├── _layer_norm_bwd_dwdb.o
#         ├── _layer_norm_bwd_dx_fused_launcher.o
#         ├── _layer_norm_bwd_dx_fused.o
#         ├── _layer_norm_fwd_fused_launcher.o
#         ├── _layer_norm_fwd_fused.o
#         └── support.o

# Parse command line options
while [ $# -gt 0 ]; do
    case $1 in
        --clean | --no-clean)
            DO_CLEAN=$1
            ;;

        --help | -h)
            help
            exit 0
            ;;

        ?*)
            echo "Invalid options:\"$1\", try $0 --help for help"
            exit 1
            ;;
      esac

      # Process next command-line option
      shift
done

if [ "x$DO_CLEAN" = "x--clean" ]; then
    echo "Cleaning build directories"
    rm -rf $BUILD_DIR
    rm -rf ~/.triton
fi

### TODO: Options for build function
# 1. build
# 2. copy shape config


C_KERNELS=""
TRITON_KERNELS=""
DRIVERS=""

# Iterate over each entry and build the driver
for entry in "${drivers[@]}"; do
  # Read the three components into variables
  IFS=' ' read -r c_kernel triton_kernel driver_path <<< "$entry"

  # Set environment variables
  C_KERNELS+=" ${c_kernel}"
  TRITON_KERNELS+=" ${triton_kernel}"
  DRIVERS+=" ${driver_path}"
done

# Optionally export them if build_triton_driver requires
export C_KERNELS
export TRITON_KERNELS
export DRIVERS

# Add ztc module to the PYTHONPATH, so that python can import ztc plugins
export PYTHONPATH=${TRITON_PLUGIN_DIRS}/python:${PYTHONPATH}
# Avoid cache the compiled kernel
export TRITON_ALWAYS_COMPILE=1

echo "C_KERNELS : "${C_KERNELS}
echo "TRITON_KERNELS : "${TRITON_KERNELS}
echo "Drivers : "${DRIVERS}

echo "build golden using gcc"
build_driver gcc

echo "build golden using zcc"
build_driver zcc

echo "build triton kernel"
build_driver triton
