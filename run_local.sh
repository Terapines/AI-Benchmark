#!/bin/bash

DIR=`dirname $0`
SRC_DIR=${DIR}/src
BUILD_DIR=${DIR}/build
KERNEL_LAUNCHER_INCLUDE_DIR=${BUILD_DIR}/aux/include

PYC="python"

ZCC="clang++-19 -g -std=c++17 -fno-lto -fvectorize -fslp-vectorize -I ${DIR}/include -I ${KERNEL_LAUNCHER_INCLUDE_DIR}"
AR="llvm-ar-19"

C_KERNELS=(
  # ${SRC_DIR}/c/correlation.cpp
  # ${SRC_DIR}/c/layernorm.cpp
  # ${SRC_DIR}/c/matmul.cpp
  # ${SRC_DIR}/c/softmax.cpp
  # ${SRC_DIR}/c/rope.cpp
  # ${SRC_DIR}/c/dropout.cpp
  ${SRC_DIR}/c/resize.cpp
  # ${SRC_DIR}/c/warp.cpp
)

# TRITON_KERNELS=`ls ${SRC_DIR}/triton/*.py`
TRITON_KERNELS=(
  # ${SRC_DIR}/triton/correlation.py
  # ${SRC_DIR}/triton/layernorm.py
  # ${SRC_DIR}/triton/matmul.py
  # ${SRC_DIR}/triton/softmax.py
  # ${SRC_DIR}/triton/rope.py
  # ${SRC_DIR}/triton/dropout.py
  ${SRC_DIR}/triton/resize.py
  # ${SRC_DIR}/triton/warp.py
)

# DRIVERS=`ls ${SRC_DIR}/main/*.cpp`
DRIVERS=(
  # ${SRC_DIR}/main/correlation.cpp
  # ${SRC_DIR}/main/layernorm.cpp
  # ${SRC_DIR}/main/matmul.cpp
  # ${SRC_DIR}/main/softmax_kernel.cpp
  # ${SRC_DIR}/main/rope.cpp
  # ${SRC_DIR}/main/dropout.cpp
  ${SRC_DIR}/main/resize.cpp
  # ${SRC_DIR}/main/warp.cpp
)

# build support library
build_support_lib() {
  ${ZCC} -O3 -c ${SRC_DIR}/support/*.cpp -o ${BUILD_DIR}/obj/support/support.o
  ${AR} rcs ${BUILD_DIR}/lib/libsupport.a ${BUILD_DIR}/obj/support/support.o
}

# build c kernel
build_c_kernel_lib() {
  for kernel in ${C_KERNELS[@]}; do
    name=`basename ${kernel} .cpp`
    ${ZCC} -O3 -c ${kernel} -o ${BUILD_DIR}/obj/c/${name}.o
  done
  find ${BUILD_DIR}/obj/c/ -name "*.o" | xargs ${AR} rcs ${BUILD_DIR}/lib/libckernel.a
}

# build triton kernel
build_triton_kernel_lib() {
  # Python virtual environment for triton kernel compilation
  TRITON_PLUGIN_DIRS=~/triton-cpu/
  TRITON_PYTHON_VENV=${TRITON_PLUGIN_DIRS}/.venv
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

      ${ZCC} -O3 -S -x ir ${kernel_ir} -o ${KERNEL_AUX_FILE_DIR}/${kernel_name}.s

      ${ZCC} -O3 -c -o ${BUILD_DIR}/obj/triton/${kernel_name}.o ${KERNEL_AUX_FILE_DIR}/${kernel_name}.s
    done

    # build triton laucher: launcher.cpp --> .o
    for kernel_launcher in ${KERNEL_AUX_FILE_DIR}/*.cpp; do
      launcher_name=`basename ${kernel_launcher} .cpp`
      ${ZCC} -O3 -c ${kernel_launcher} -fopenmp -o ${BUILD_DIR}/obj/triton/${launcher_name}.o
    done

  done

  find ${BUILD_DIR}/obj/triton/ -name "*.o" | xargs ${AR} rcs ${BUILD_DIR}/lib/libtritonkernel.a
}

# build driver
build_driver(){
  for main in ${DRIVERS[@]}; do
    name=`basename ${main} .cpp`

    KERNEL_BIN_DIR=${BUILD_DIR}/bin/${name}/
    mkdir -p ${BUILD_DIR}/bin/${name}

    # Compile driver
    # .elf suffix to avoid scp problem(same name dir and kernel)
    ${ZCC} -O3 ${main} -L ${BUILD_DIR}/lib -fopenmp -lckernel -ltritonkernel -lsupport -fPIC -DC_KERNEL_ENABLE -o ${KERNEL_BIN_DIR}/${name}_c.elf
    ${ZCC} -O3 ${main} -L ${BUILD_DIR}/lib -fopenmp -lckernel -ltritonkernel -lsupport -fPIC -DTRITON_KERNEL_ENABLE -o ${KERNEL_BIN_DIR}/${name}_triton.elf
    # Data shape config
    cp ${SRC_DIR}/main/${name}.cfg  ${KERNEL_BIN_DIR}
  done
}

run(){
  # remove all .bin files if exist
  # rm -f ./*.bin
  BENCHMARK=${BUILD_DIR}/bin/

  for f_sub in `ls ${BENCHMARK}`; do
    ### FIXME: Check whether is a kernel directory
    kernel_dir=${BENCHMARK}/${f_sub}
    echo "${kernel_dir}"
    if [ ! -d "${kernel_dir}" ];then
        continue
    fi

    kernel_name=`basename ${f_sub}`
    echo ${kernel_name}

    # shape array
    # NOTE: get from config
    source ${kernel_dir}/${kernel_name}.cfg
    echo ${SHAPE[*]}

    for shape in ${SHAPE[@]}; do
      DB_FILE=${DIR}/${kernel_name} ${kernel_dir}/${kernel_name}_c.elf ${shape}
      DB_FILE=${DIR}/${kernel_name} ${kernel_dir}/${kernel_name}_triton.elf ${shape}
    done
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
run