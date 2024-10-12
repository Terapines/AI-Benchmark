#!/bin/bash

DIR=$(cd "$(dirname "$0")"; pwd)
echo ${DIR}

SRC_DIR=${DIR}/src
BUILD_DIR=${DIR}/build

# Always check accuracy
MODE="Accuracy"
# Update run.sh mode
sed -i "s/MODE=\(\".*\"\)/MODE=\"${MODE}\"/g" ${DIR}/run.sh
sed -i "s/MODE=\(\".*\"\)/MODE=\"${MODE}\"/g" ${DIR}/report.sh

# C compile env
ARCH=rv64gcv_zvl256b
ABI=lp64d

ZCC="z++ -fno-lto --target=riscv64-unknown-linux-gnu -march=${ARCH} -mabi=${ABI} -O3"
AR="llvm-ar"
# OBJDUMP="llvm-objdump"

# Python virtual environment for triton kernel compilation
PYC="python"
TRITON_PLUGIN_DIRS=~/workspace/AI-Kernel-Library/triton-cpu/
TRITON_PYTHON_VENV=${TRITON_PLUGIN_DIRS}/.venv

KERNEL_LAUNCHER_INCLUDE_DIR=${BUILD_DIR}/aux/include/

# Compilation threads
MAX_MULTITHREADING=8

# Default clean build directory
DO_CLEAN="--clean"


### FIXME: Choose which kernels should be compiled
# FIXME: Use config

# Array of "kernel_path driver_path tunning_arg" entries
drivers=(
  "triton/layernorm.py main/layernorm.cpp _layer_norm_fwd_fused"
  "triton/layernorm.py main/layernorm.cpp _layer_norm_bwd_fused"
  "triton/correlation.py main/correlation.cpp correlation_kernel"
  "triton/softmax.py main/softmax_kernel.cpp softmax_kernel"
  "triton/matmul.py main/matmul.cpp matmul_kernel"
  "triton/rope.py main/rope.cpp rope_kernel"
  "triton/dropout.py main/dropout.cpp dropout_kernel"
  "triton/resize.py main/resize.cpp resize_kernel"
  "triton/warp.py main/warp.cpp warp_kernel"
)


# Helper function
help()
{
cat <<END
Build AI-Benchmark.

Usage: ./autotuning.sh [--clean | --no-clean]
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

create_dir_hierarchy(){
  rm -rf ${LIB_DIR}
  rm -rf ${BIN_DIR}
  rm -rf ${OBJ_DIR}
  mkdir -p ${LIB_DIR}
  mkdir -p ${BIN_DIR}
  mkdir -p ${OBJ_DIR}
}

# build triton kernel
# build_triton_driver + ${ENABLE_AUTOTUNING}
build_triton_driver() {
  source ${TRITON_PYTHON_VENV}/bin/activate

  echo ${TRITON_KERNEL}
  name=`basename ${TRITON_KERNEL} .py`

  ### FIXME: Modified triton-cpu to generate these files to the BUILD_DIR direcly
  # triton-cpu kernel launcher

  KERNEL_AUX_FILE_DIR=${BUILD_DIR}/aux/src/$1
  echo ${KERNEL_AUX_FILE_DIR}

  mkdir -p ${KERNEL_LAUNCHER_INCLUDE_DIR}
  mkdir -p ${KERNEL_AUX_FILE_DIR}

  # compile triton kernel: .py --> .llir + launcher.cpp
  # TRITON_ALWAYS_COMPILE=1 MLIR_ENABLE_DUMP=1
  ### FIXME: How to specify which kernel to enable among multiple kernels, and whether to enable them simultaneously
  ### Two parameters may be needed to control, one parameter controls ENABLE, and the other parameter controls which ENABLE.
  ENABLE_AUTOTUNING=$1 KERNEL_LAUNCHER_INCLUDE_DIR=${KERNEL_LAUNCHER_INCLUDE_DIR} KERNEL_AUX_FILE_DIR=${KERNEL_AUX_FILE_DIR} ${PYC} ${TRITON_KERNEL}

  driver_name=`basename ${DRIVER} .cpp`
  echo ${DRIVER}

  KERNEL_BIN_DIR=${BIN_DIR}/${driver_name}/
  mkdir -p ${KERNEL_BIN_DIR}

  # Data shape config
  cp ${SRC_DIR}/main/${driver_name}.cfg  ${KERNEL_BIN_DIR}


  # multi-thread
  [ -e /tmp/fd1_hyc ] || mkfifo /tmp/fd1_hyc
  exec 6<>/tmp/fd1_hyc
  rm -rf /tmp/fd1_hyc

  for ((i=1;i<=$MAX_MULTITHREADING;i++))
  do
      echo >&6
  done

  for tunning_dir in ${KERNEL_AUX_FILE_DIR}_*; do
    read -u6
    {

    echo "----------${tunning_dir}-------------"
    block_shape=${tunning_dir#*$1_}
    mkdir -p ${OBJ_DIR}/${name}_$1_${block_shape}

    # soft link common kernel llir file
    find "${KERNEL_AUX_FILE_DIR}" -maxdepth 1 -type f -exec ln -s {} "${tunning_dir}" \;

    # TODO: Update Clang version
    # For now, we just replace the trunc n[us]w with trunc
    sed -i 's/trunc nuw nsw/trunc/g; s/trunc nuw/trunc/g; s/trunc nsw/trunc/g' ${tunning_dir}/*.llir

    # build triton kernel: .llir --> .o
    for kernel_ir in ${tunning_dir}/*.llir; do
      kernel_name=`basename ${kernel_ir} .llir`
      echo ${kernel_ir}

      ${ZCC} -S -x ir ${kernel_ir} -mllvm --riscv-disable-rvv-fixedlen=false -mrvv-vector-bits=256 -o ${KERNEL_AUX_FILE_DIR}_${block_shape}/${kernel_name}.s

      ${ZCC} -c -o ${OBJ_DIR}/${name}_$1_${block_shape}/${kernel_name}.o ${KERNEL_AUX_FILE_DIR}_${block_shape}/${kernel_name}.s
    done

    for kernel_launcher in ${tunning_dir}/*.cpp; do
      launcher_name=`basename ${kernel_launcher} .cpp`
      ${ZCC} -I ${DIR}/include -I ${KERNEL_LAUNCHER_INCLUDE_DIR} -c ${kernel_launcher} -fopenmp -o ${OBJ_DIR}/${name}_$1_${block_shape}/${launcher_name}.o
    done


    find ${OBJ_DIR}/${name}_$1_${block_shape}/ -not -name "support.o"  -name "*.o" | xargs ${AR} rcs ${LIB_DIR}/libkernel_$1_${block_shape}.a

    # Compile driver
    # .elf suffix to avoid scp problem(same name dir and kernel)
    # Always check accurary
    ${COMPILER} ${DRIVER} -I ${DIR}/include -I ${KERNEL_LAUNCHER_INCLUDE_DIR} -L ${LIB_DIR} -fopenmp -lkernel_$1_${block_shape} -lsupport -latomic -std=c++17 -D${KERNEL_ENABLE} -DCHECK_ACCURACY -fPIC -o ${KERNEL_BIN_DIR}/${driver_name}_$1_${block_shape}.elf

    # ${OBJDUMP} -d ${KERNEL_BIN_DIR}/${driver_name}_$1_${block_shape}.elf &> ${KERNEL_BIN_DIR}/${driver_name}_$1_${block_shape}.elf.s

      echo >&6
    } &
  done
  wait
  exec 6>&-
}

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

### TODO: Options for build function
# 1. build
# 2. copy shape config

# build triton driver
COMPILER=${ZCC}
LIB_DIR=${BUILD_DIR}/lib/triton
BIN_DIR=${BUILD_DIR}/bin/triton
OBJ_DIR=${BUILD_DIR}/obj/triton
KERNEL_ENABLE=TRITON_KERNEL_ENABLE

if [ "x$DO_CLEAN" = "x--clean" ]; then
    echo "Cleaning triton build directories"
    rm -rf $BIN_DIR
    rm -rf $LIB_DIR
    rm -rf $OBJ_DIR
    rm -rf ${BUILD_DIR}/aux/
fi

create_dir_hierarchy
mkdir -p ${BUILD_DIR}/aux/src

build_support_lib


# Iterate over each entry and build the driver
for entry in "${drivers[@]}"; do
  # Read the three components into variables
  IFS=' ' read -r kernel_path driver_path tunning_arg <<< "$entry"

  # Set environment variables
  TRITON_KERNEL="${SRC_DIR}/${kernel_path}"
  DRIVER="${SRC_DIR}/${driver_path}"

  # Optionally export them if build_triton_driver requires
  export TRITON_KERNEL
  export DRIVER

  # Call the build function with the specified argument
  build_triton_driver "$tunning_arg"

  # Unset variables if they shouldn't persist
  unset TRITON_KERNEL
  unset DRIVER
done