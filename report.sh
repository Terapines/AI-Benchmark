#!/bin/bash

DIR=`dirname $0`

BENCHMARK=${DIR}/build/bin

REPORT_FILE=${DIR}/build/report.xls

MODE="Accuracy"

ARCH=rv64gcv
ABI=lp64d

GCC="riscv64-unknown-linux-gnu-g++ -march=${ARCH} -mabi=${ABI} -O3 -fopenmp -fPIC"
ZCC="z++ -fno-lto --target=riscv64-unknown-linux-gnu -march=${ARCH} -mabi=${ABI} -O3 -fopenmp -fPIC"


# | #####  softmax_kernel kernel performance ##### |
# | shape (RxCxRUN_COUNT) | gcc_T1  | zcc_T1  | triton_T1 | gcc_T4  | zcc_T4  | triton_T4 | gcc_T8   | zcc_T8   | triton_T8 |
# | --------------------- | ------- | ------- | --------- | ------- | ------- | --------- | -------- | -------- | --------- |
# | 512x2048x100          | 5.24323 | 5.10248 | 5.17715   | 1.35107 | 1.34665 | 1.29518   | 0.729931 | 0.711579 | 0.682701  |
# | 512x2560x100          | 6.56828 | 6.59798 | 6.33971   | 1.67922 | 1.67726 | 1.63183   | 0.924294 | 0.912023 | 0.856632  |


echo "Report performace to ${REPORT_FILE}"

# Keyword to extract the kernel running time
# STAT_KEYWORD=(C Triton)

COMPILER=(gcc zcc triton)
THREADS=(1 4 8)

TRITON_KERNELS=`ls ${BENCHMARK}/triton/`
# TRITON_KERNELS=layernorm

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
  for thread in ${THREADS[@]}; do
    for compiler in ${COMPILER[@]}; do
      for kernel in `ls -v ${BENCHMARK}/${compiler}/${kernel_name}/${kernel_name}*.elf`; do
        tmp=`basename ${kernel} .elf`
        block_shape=${tmp#${kernel_name}*}
        echo -ne "\t${compiler}_T${thread}${block_shape}" >> ${REPORT_FILE}
      done
    done
  done
  echo "" >> ${REPORT_FILE}

  # average_percentage=0.0
  for shape in ${SHAPE[@]}; do
    echo -ne "${shape}" >> ${REPORT_FILE}

    for thread in ${THREADS[@]}; do
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
        for kernel in `ls -v ${kernel_dir}/${kernel_name}*.elf`; do
          echo ${kernel}
          tmp=`basename ${kernel} .elf`

          second=$(cat ${kernel_dir}/${tmp}_T${thread}_S${shape}.log | sed -n "s/^.* Kernel Time: \([0-9]\+\(\.[0-9]\+\)*\).*/\1/p")
          # percentage=$(echo "scale=2; ${second} / ${percentage}" | bc)

          echo -ne "\t${second}" >> ${REPORT_FILE}
        done
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
