#!/bin/bash

DIR=`dirname $0`

BENCHMARK=${DIR}/bin/

THREAD=(1 4 8)

# COMPILER=`ls ${BENCHMARK}`
COMPILER=(triton gcc zcc)

for compiler in ${COMPILER[@]}; do
  for f_sub in `ls ${BENCHMARK}/${compiler}`; do
    ### FIXME: Check whether is a kernel directory
    kernel_dir=${BENCHMARK}/${compiler}/${f_sub}
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

    for thread in ${THREAD[@]}; do
      for shape in ${SHAPE[@]}; do
        DB_FILE=${DIR}/${kernel_name} TRITON_CPU_MAX_THREADS=${thread} ${kernel_dir}/${kernel_name}.elf ${shape} 2> ${kernel_dir}/${kernel_name}_T${thread}_S${shape}.log
      done
    done

  done
done