#!/bin/bash

DIR=`dirname $0`
SRC_DIR=${DIR}/src
BUILD_DIR=${DIR}/build
KERNEL_LAUNCHER_INCLUDE_DIR=${BUILD_DIR}/aux/include

PYC="python"

ARCH=rv64gcv
ABI=lp64d
ZCC="z++ -fno-lto --target=riscv64-unknown-linux-gnu -march=${ARCH} -mabi=${ABI} -O3 -I ${DIR}/include  -I ${KERNEL_LAUNCHER_INCLUDE_DIR} "
AR="llvm-ar"

# build support library
build_support_lib() {
  ${ZCC} -c ${SRC_DIR}/support/*.cpp -o ${BUILD_DIR}/obj/support/support.o
  ${AR} rcs ${BUILD_DIR}/lib/libsupport.a ${BUILD_DIR}/obj/support/support.o
}

# build c kernel
build_c_kernel_lib() {
  for kernel in ${SRC_DIR}/c/*.cpp; do
    name=`basename ${kernel} .cpp`
    ${ZCC} -c ${kernel} -o ${BUILD_DIR}/obj/c/${name}.o
  done
  find ${BUILD_DIR}/obj/c/ -name "*.o" | xargs ${AR} rcs ${BUILD_DIR}/lib/libckernel.a
}

# build triton kernel
build_triton_kernel_lib() {
  # Python virtual environment for triton kernel compilation
  TRITON_PLUGIN_DIRS=~/workspace/AI-Kernel-Library/triton-cpu/
  TRITON_PYTHON_VENV=${TRITON_PLUGIN_DIRS}/.venv
  source ${TRITON_PYTHON_VENV}/bin/activate

  for kernel in ${SRC_DIR}/triton/*.py; do
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

      ${ZCC} -c -o ${BUILD_DIR}/obj/triton/${kernel_name}.o ${KERNEL_AUX_FILE_DIR}/${kernel_name}.s
    done

    # build triton laucher: launcher.cpp --> .o
    for kernel_launcher in ${KERNEL_AUX_FILE_DIR}/*.cpp; do
      launcher_name=`basename ${kernel_launcher} .cpp`
      ${ZCC} -c ${kernel_launcher} -fopenmp -o ${BUILD_DIR}/obj/triton/${launcher_name}.o
    done

  done

  find ${BUILD_DIR}/obj/triton/ -name "*.o" | xargs ${AR} rcs ${BUILD_DIR}/lib/libtritonkernel.a
}

# build driver
build_driver(){
  for main in ${SRC_DIR}/main/*.cpp; do
    name=`basename ${main} .cpp`

    KERNEL_BIN_DIR=${BUILD_DIR}/bin/${name}/
    mkdir -p ${BUILD_DIR}/bin/${name}

    # Compile driver
    # .elf suffix to avoid scp problem(same name dir and kernel)
    ${ZCC} ${main} -L ${BUILD_DIR}/lib -fopenmp -lckernel -ltritonkernel -latomic -lsupport -fPIC -o ${KERNEL_BIN_DIR}/${name}.elf
    # Data shape config
    cp ${SRC_DIR}/main/${name}.cfg  ${KERNEL_BIN_DIR}
  done
}


# build
# ├── aux    // When use MAGIC compiler, we may generate *.o or lib directly.
# │   ├── include
# │   │   └── layernorm_launcher.h
# │   └── src
# │       └── layernorm
# │           ├── _layer_norm_bwd_dwdb.llir
# │           ├── _layer_norm_bwd_dwdb.s
# │           ├── _layer_norm_bwd_dx_fused.llir
# │           ├── _layer_norm_bwd_dx_fused.s
# │           ├── _layer_norm_fwd_fused.llir
# │           ├── _layer_norm_fwd_fused.s
# │           └── layernorm_launcher.cpp
# ├── bin
# │   └── layernorm
# │       ├── layernorm.cfg
# │       └── layernorm.elf
# ├── lib
# │   ├── libckernel.a
# │   ├── libsupport.a
# │   └── libtritonkernel.a
# └── obj
#     ├── c
#     │   └── layernorm.o
#     ├── support
#     │   └── support.o
#     └── triton
#         ├── _layer_norm_bwd_dwdb.o
#         ├── _layer_norm_bwd_dx_fused.o
#         ├── _layer_norm_fwd_fused.o
#         └── layernorm_launcher.o


rm -rf ${BUILD_DIR}
mkdir -p ${BUILD_DIR}/aux/include
mkdir -p ${BUILD_DIR}/aux/src
mkdir -p ${BUILD_DIR}/bin
mkdir -p ${BUILD_DIR}/lib
mkdir -p ${BUILD_DIR}/obj/c
mkdir -p ${BUILD_DIR}/obj/support
mkdir -p ${BUILD_DIR}/obj/triton

build_support_lib
build_c_kernel_lib
build_triton_kernel_lib
build_driver