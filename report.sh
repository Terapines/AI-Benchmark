#!/bin/bash

DIR=`dirname $0`

BENCHMARK=${DIR}/bin/

REPORT_FILE=${BENCHMARK}/report.xls


ARCH=rv64gcv
ABI=lp64d

ZCC="zcc -fno-lto --target=riscv64-unknown-linux-gnu -march=${ARCH} -mabi=${ABI} -O3 -fopenmp=libgomp -fPIC -static "


# #####  kernel performance #####
# | shape	    |   c	    | triton	| triton/c |
# | --------- |---------|---------|--------- |
# | 1151x8192	| 0.31966 | 4.26706 |	13.76    |
# | 1151x8192	| 0.31966	| 4.26706	| 13.76    |
# | 1151x8192	| 0.31966	| 4.26706	| 13.76    |
# |average percentage	|	|         | 13.76    |



echo "Report performace to ${REPORT_FILE}"

# Keyword to extract the kernel running time
STAT_KEYWORD=(C Triton)

echo -n "" > ${REPORT_FILE}
# Kernel performance on different shape
for f_sub in `ls ${BENCHMARK}`; do
  ### FIXME: Check whether is a kernel directory
  kernel_dir=${BENCHMARK}/${f_sub}
  echo "${kernel_dir}"
  if [ ! -d "${kernel_dir}" ];then
      continue
  fi


  kernel_name=`basename ${f_sub}`

  echo "${kernel_name}"

  # header
  echo -e "##### ${kernel_name} kernel performance #####" >> ${REPORT_FILE}
  echo -e "shape\tc\ttriton\ttriton/c" >> ${REPORT_FILE}

  # shape array
  # NOTE: get from config
  source ${kernel_dir}/${kernel_name}.cfg
  echo ${SHAPE[*]}

  average_percentage=0.0
  for shape in ${SHAPE[@]}; do
    echo -ne "${shape}" >> ${REPORT_FILE}

    #=================================================#
    # NOTE: depend on the format of perf.log
    # extract the statistics
    percentage=1.0
    for tmp in ${STAT_KEYWORD[@]}; do

      second=$(cat ${kernel_dir}/${kernel_name}_${shape}.log | sed -n "s/^.*${tmp} Kernel Time: \([0-9]\+\(\.[0-9]\+\)*\).*/\1/p")

      percentage=$(echo "scale=2; ${second} / ${percentage}" | bc)

      echo -ne "\t${second}" >> ${REPORT_FILE}
    done
    #=================================================#

    # calculate the performance gap percentage
    echo -ne "\t${percentage}" >> ${REPORT_FILE}

    # Accumulate performance gap percentage
    average_percentage=$(echo "${average_percentage} + ${percentage}" | bc)

    echo "" >> ${REPORT_FILE}
  done
  # Average performance gap percentage
  average_percentage=$(echo "scale=2; ${average_percentage}/${#SHAPE[@]} " | bc)

  tabs=$(printf '\t%.0s' $(seq 1 ${#STAT_KEYWORD[@]}))
  echo -e "average percentage\t${tabs}${average_percentage}" >> ${REPORT_FILE}

  echo "" >> ${REPORT_FILE}
  echo "" >> ${REPORT_FILE}
done


echo "" >> ${REPORT_FILE}
echo "" >> ${REPORT_FILE}


# May add triton-cpu version?

echo "${ZCC}" >> ${REPORT_FILE}
# ${ZCC} --version >> ${REPORT_FILE}
echo "" >> ${REPORT_FILE}