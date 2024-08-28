#!/bin/bash

DIR=`dirname $0`

BENCHMARK=${DIR}/bin/

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
    TRITON_CPU_OMP_DEBUG=1 ${kernel_dir}/${kernel_name}.elf ${shape} 2> ${kernel_dir}/${kernel_name}_${shape}.log
  done
done