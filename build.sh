#!/bin/bash

DIR=`dirname $0`
SRC_DIR=${DIR}/src
BUILD_DIR=${DIR}/build

# C compile env
ARCH=rv64gcv
ABI=lp64d

GCC="riscv64-unknown-linux-gnu-g++ -march=${ARCH} -mabi=${ABI} -O3"
ZCC="z++ -fno-lto --target=riscv64-unknown-linux-gnu -march=${ARCH} -mabi=${ABI} -O3"
AR="llvm-ar"

# Python virtual environment for triton kernel compilation
PYC="python"
TRITON_PLUGIN_DIRS=~/workspace/AI-Kernel-Library/triton-cpu/
TRITON_PYTHON_VENV=${TRITON_PLUGIN_DIRS}/.venv
# triton-cpu kernel launcher
KERNEL_LAUNCHER_INCLUDE_DIR=${BUILD_DIR}/aux/include


### FIXME: Choose which kernels should be compiled
# C_KERNELS=`ls ${SRC_DIR}/c/*.cpp`
C_KERNELS=(
  ${SRC_DIR}/c/correlation.cpp
  ${SRC_DIR}/c/layernorm.cpp
  ${SRC_DIR}/c/matmul.cpp
  ${SRC_DIR}/c/softmax.cpp
  ${SRC_DIR}/c/rope.cpp
  ${SRC_DIR}/c/dropout.cpp
)

# TRITON_KERNELS=`ls ${SRC_DIR}/triton/*.py`
TRITON_KERNELS=(
  ${SRC_DIR}/triton/correlation.py
  ${SRC_DIR}/triton/layernorm.py
  ${SRC_DIR}/triton/matmul.py
  ${SRC_DIR}/triton/softmax.py
  ${SRC_DIR}/triton/rope.py
  ${SRC_DIR}/triton/dropout.py
)

# DRIVERS=`ls ${SRC_DIR}/main/*.cpp`
DRIVERS=(
  ${SRC_DIR}/main/correlation.cpp
  ${SRC_DIR}/main/layernorm.cpp
  ${SRC_DIR}/main/matmul.cpp
  ${SRC_DIR}/main/softmax_kernel.cpp
  ${SRC_DIR}/main/rope.cpp
  ${SRC_DIR}/main/dropout.cpp
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
    ${COMPILER} -fPIC -I ${DIR}/include -c ${kernel} -fopenmp -o ${OBJ_DIR}/${name}.o
  done

  find ${OBJ_DIR} -not -name "support.o" -name "*.o" | xargs ${AR} rcs ${LIB_DIR}/libkernel.a
}

# build triton kernel
build_triton_kernel_lib() {
  source ${TRITON_PYTHON_VENV}/bin/activate

  for kernel in ${TRITON_KERNELS[@]}; do
    name=`basename ${kernel} .py`

    ### FIXME: Modified triton-cpu to generate these files to the BUILD_DIR direcly
    KERNEL_AUX_FILE_DIR=${BUILD_DIR}/aux/src/${name}/
    mkdir -p ${KERNEL_AUX_FILE_DIR}

    echo ${kernel}
    # compile triton kernel: .py --> .llir + launcher.cpp
    KERNEL_LAUNCHER_INCLUDE_DIR=${KERNEL_LAUNCHER_INCLUDE_DIR} KERNEL_AUX_FILE_DIR=${KERNEL_AUX_FILE_DIR} ${PYC} ${kernel}

    # build triton kernel: .llir --> .o
    for kernel_ir in ${KERNEL_AUX_FILE_DIR}/*.llir; do
      kernel_name=`basename ${kernel_ir} .llir`
      echo ${kernel_ir}
      # llc -march=riscv64 -mattr=+d,v  ${kernel_ir} -o ${KERNEL_AUX_FILE_DIR}/${kernel_name}.s
      # z++ -march=rv64gcv -fno-lto --target=riscv64-unknown-linux-gnu -S -x ir  -O2 ${kernel_ir} -mllvm --riscv-disable-rvv-fixedlen=false -mrvv-vector-bits=256 -o ${KERNEL_AUX_FILE_DIR}/${kernel_name}.s

      ${ZCC} -S -x ir ${kernel_ir} -mllvm --riscv-disable-rvv-fixedlen=false -mrvv-vector-bits=256 -o ${KERNEL_AUX_FILE_DIR}/${kernel_name}.s

      ${ZCC} -c -o ${OBJ_DIR}/${kernel_name}.o ${KERNEL_AUX_FILE_DIR}/${kernel_name}.s
    done

    # build triton laucher: launcher.cpp --> .o
    for kernel_launcher in ${KERNEL_AUX_FILE_DIR}/*.cpp; do
      launcher_name=`basename ${kernel_launcher} .cpp`
      ${ZCC} -I ${DIR}/include -I ${KERNEL_LAUNCHER_INCLUDE_DIR} -c ${kernel_launcher} -fopenmp -o ${OBJ_DIR}/${launcher_name}.o
    done

  done

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

  for main in ${DRIVERS[@]}; do
    name=`basename ${main} .cpp`
    echo ${main}

    KERNEL_BIN_DIR=${BIN_DIR}/${name}/
    mkdir -p ${KERNEL_BIN_DIR}

    # Compile driver
    # .elf suffix to avoid scp problem(same name dir and kernel)
    ${COMPILER} ${main} -I ${DIR}/include -I ${KERNEL_LAUNCHER_INCLUDE_DIR} -L ${LIB_DIR} -fopenmp -lkernel -lsupport -latomic -std=c++17 -D${KERNEL_ENABLE} -fPIC -o ${KERNEL_BIN_DIR}/${name}.elf

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
fi

echo "C_KERNELS : "${C_KERNELS}
echo "TRITON_KERNELS : "${TRITON_KERNELS}
echo "Drivers : "${DRIVERS}

### TODO: Options for build function
# 1. build
# 2. copy shape config
echo "build golden using gcc"
build_driver gcc

echo "build golden using zcc"
build_driver zcc

echo "build triton kernel"
build_driver triton