#!/bin/bash

DIR=`dirname $0`

BENCHMARK=${DIR}/build/bin

REPORT_FILE=${DIR}/build/report.xls


ARCH=rv64gcv
ABI=lp64d

GCC="riscv64-unknown-linux-gnu-g++ -march=${ARCH} -mabi=${ABI} -O3 -fopenmp -fPIC"
ZCC="z++ -fno-lto --target=riscv64-unknown-linux-gnu -march=${ARCH} -mabi=${ABI} -O3 -fopenmp -fPIC"


# | #####  softmax_kernel kernel performance ##### |
# | shape (RxCxRUN_COUNT)                   | gcc_T1  | gcc_T2  | gcc_T4   | gcc_T8   | zcc_T1  | zcc_T2  | zcc_T4   | zcc_T8   | triton_T1 | triton_T2 | triton_T4 | triton_T8 |
# | --------------------------------------- | ------- | ------- | -------- | -------- | ------- | ------- | -------- | -------- | --------- | --------- | --------- | --------- |
# | 1151x8192x10                            | 5.16791 | 2.6222  | 1.30873  | 0.676481 | 5.14242 | 2.59087 | 1.30462  | 0.672324 | 0.937089  | 0.500839  | 0.261468  | 0.141019  |
# | 1151x4096x10                            | 2.58569 | 1.30287 | 0.657147 | 0.340676 | 2.5699  | 1.30141 | 0.656113 | 0.341305 | 0.929977  | 0.490913  | 0.252563  | 0.140692  |
# | 1151x2048x10                            | 1.29152 | 0.65242 | 0.334033 | 0.170301 | 1.29245 | 0.65099 | 0.328486 | 0.173065 | 0.920868  | 0.488619  | 0.252841  | 0.138696  |


echo "Report performace to ${REPORT_FILE}"

# Keyword to extract the kernel running time
STAT_KEYWORD=(C Triton)

COMPILER=(gcc zcc triton)
THREADS=(1 2 4 8)

TRITON_KERNELS=`ls ${BENCHMARK}/triton/`

echo -n "" > ${REPORT_FILE}
# Kernel performance on different shape
for kernel_name in ${TRITON_KERNELS}; do
  echo "${kernel_name}"
  # header
  # shape array
  # NOTE: get from config
  source ${BENCHMARK}/triton/${kernel_name}/${kernel_name}.cfg
  echo ${SHAPE[*]}
  echo -e "##### ${kernel_name} kernel performance #####" >> ${REPORT_FILE}

  echo -ne "shape (${SHAPE_DESC})" >> ${REPORT_FILE}
  for compiler in ${COMPILER[@]}; do
    for thread in ${THREADS[@]}; do
      echo -ne "\t${compiler}_T${thread}" >> ${REPORT_FILE}
    done
  done
  echo "" >> ${REPORT_FILE}

  # average_percentage=0.0
  for shape in ${SHAPE[@]}; do
    echo -ne "${shape}" >> ${REPORT_FILE}

    for compiler in ${COMPILER[@]}; do
      ### FIXME: Check whether is a kernel directory
      kernel_dir=${BENCHMARK}/${compiler}/${kernel_name}
      if [ ! -d "${kernel_dir}" ];then
          continue
      fi
      echo "${kernel_dir}"

      #=================================================#
      # NOTE: depend on the format of perf.log
      # extract the statistics

      # percentage=1.0
      for thread in ${THREADS[@]}; do
        second=$(cat ${kernel_dir}/${kernel_name}_T${thread}_S${shape}.log | sed -n "s/^.* Kernel Time: \([0-9]\+\(\.[0-9]\+\)*\).*/\1/p")
        # percentage=$(echo "scale=2; ${second} / ${percentage}" | bc)

        echo -ne "\t${second}" >> ${REPORT_FILE}
      done
      #=================================================#

      # calculate the performance gap percentage
      # echo -ne "\t${percentage}" >> ${REPORT_FILE}

      # Accumulate performance gap percentage
      # average_percentage=$(echo "${average_percentage} + ${percentage}" | bc)
    done
    echo "" >> ${REPORT_FILE}

  done
  echo "" >> ${REPORT_FILE}
  # Average performance gap percentage
  # average_percentage=$(echo "scale=2; ${average_percentage}/${#SHAPE[@]} " | bc)

  # tabs=$(printf '\t%.0s' $(seq 1 ${#STAT_KEYWORD[@]}))
  # echo -e "average percentage\t${tabs}${average_percentage}" >> ${REPORT_FILE}

  echo "" >> ${REPORT_FILE}
  echo "" >> ${REPORT_FILE}
done


echo "" >> ${REPORT_FILE}
echo "" >> ${REPORT_FILE}


# May add triton-cpu version?

echo "${GCC}" >> ${REPORT_FILE}
${GCC} --version >> ${REPORT_FILE}
echo "" >> ${REPORT_FILE}

echo "${ZCC}" >> ${REPORT_FILE}
${ZCC} --version >> ${REPORT_FILE}
echo "" >> ${REPORT_FILE}
