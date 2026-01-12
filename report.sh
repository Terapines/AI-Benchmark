#!/bin/bash

DIR=`dirname $0`
# source configuration file
source ${DIR}/config.sh

BENCHMARK=${DIR}/build/bin

REPORT_FILE=${DIR}/build/report.xls
TEMP_REPORT=${DIR}/build/report.tmp
CYCLES_REPORT=${DIR}/build/cycles.xls
TEMP_CYCLES_REPORT=${DIR}/build/cycles.tmp
INSTRET_REPORT=${DIR}/build/instret.xls
TEMP_INSTRET_REPORT=${DIR}/build/instret.tmp

MODE="Accuracy"

ARCH=rv64gcv
ABI=lp64d

GCC="${GCC_PATH} -march=${ARCH} -mabi=${ABI} -O3 -fopenmp -fPIC"
ZCC="${ZCC_PATH} -fno-lto --target=riscv64-unknown-linux-gnu -march=${ARCH} -mabi=${ABI} -O3 -fopenmp -fPIC -I${MLIR_INCLUDE_DIR}"


# | #####  softmax_kernel kernel performance ##### |
# | shape (RxCxRUN_COUNT) | gcc_T1  | zcc_T1  | triton_T1 | gcc_T4  | zcc_T4  | triton_T4 | gcc_T8   | zcc_T8   | triton_T8 |
# | --------------------- | ------- | ------- | --------- | ------- | ------- | --------- | -------- | -------- | --------- |
# | 512x2048x100          | 5.24323 | 5.10248 | 5.17715   | 1.35107 | 1.34665 | 1.29518   | 0.729931 | 0.711579 | 0.682701  |
# | 512x2560x100          | 6.56828 | 6.59798 | 6.33971   | 1.67922 | 1.67726 | 1.63183   | 0.924294 | 0.912023 | 0.856632  |


echo "Report performace to ${REPORT_FILE}"
echo "Report cycles to ${CYCLES_REPORT}"
echo "Report instret to ${INSTRET_REPORT}"

# Keyword to extract the kernel running time
# STAT_KEYWORD=(C Triton)

COMPILER=(gcc zcc triton)
THREADS=(1 4 8)

TRITON_KERNELS=`ls ${BENCHMARK}/triton/`
# TRITON_KERNELS=layernorm

echo -n "" > ${TEMP_REPORT}
echo -n "" > ${TEMP_CYCLES_REPORT}
echo -n "" > ${TEMP_INSTRET_REPORT}
# Kernel performance on different shape
for kernel_name in ${TRITON_KERNELS}; do
  echo "${kernel_name}"
  # header
  # shape array
  # NOTE: get from config
  source ${BENCHMARK}/triton/${kernel_name}/${kernel_name}.cfg
  echo ${SHAPE[*]}
  echo -e "##### ${kernel_name} kernel performance #####" >> ${TEMP_REPORT}
  echo -e "##### ${kernel_name} kernel cycles #####" >> ${TEMP_CYCLES_REPORT}
  echo -e "##### ${kernel_name} kernel instret #####" >> ${TEMP_INSTRET_REPORT}

  echo -ne "shape (${SHAPE_DESC})" >> ${TEMP_REPORT}
  echo -ne "shape (${SHAPE_DESC})" >> ${TEMP_CYCLES_REPORT}
  echo -ne "shape (${SHAPE_DESC})" >> ${TEMP_INSTRET_REPORT}
  for thread in ${THREADS[@]}; do
    for compiler in ${COMPILER[@]}; do
      for kernel in `ls -v ${BENCHMARK}/${compiler}/${kernel_name}/${kernel_name}*.elf`; do
        tmp=`basename ${kernel} .elf`
        block_shape=${tmp#${kernel_name}*}
        echo -ne "\t${compiler}_T${thread}${block_shape}" >> ${TEMP_REPORT}
        echo -ne "\t${compiler}_T${thread}${block_shape}" >> ${TEMP_CYCLES_REPORT}
        echo -ne "\t${compiler}_T${thread}${block_shape}" >> ${TEMP_INSTRET_REPORT}
      done
    done
  done
  echo "" >> ${TEMP_REPORT}
  echo "" >> ${TEMP_CYCLES_REPORT}
  echo "" >> ${TEMP_INSTRET_REPORT}

  # average_percentage=0.0
  for shape in ${SHAPE[@]}; do
    echo -ne "${shape}" >> ${TEMP_REPORT}
    echo -ne "${shape}" >> ${TEMP_CYCLES_REPORT}
    echo -ne "${shape}" >> ${TEMP_INSTRET_REPORT}

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
          cycles=$(cat ${kernel_dir}/${tmp}_T${thread}_S${shape}.log | sed -n "s/^.* Total cycles (sum): \([0-9]\+\).*/\1/p")
          instret=$(cat ${kernel_dir}/${tmp}_T${thread}_S${shape}.log | sed -n "s/^.* Total instret (sum): \([0-9]\+\).*/\1/p")
          # percentage=$(echo "scale=2; ${second} / ${percentage}" | bc)

          echo -ne "\t${second}" >> ${TEMP_REPORT}
          echo -ne "\t${cycles}" >> ${TEMP_CYCLES_REPORT}
          echo -ne "\t${instret}" >> ${TEMP_INSTRET_REPORT}
        done
      done
      #=================================================#

      # calculate the performance gap percentage
      # echo -ne "\t${percentage}" >> ${TEMP_REPORT}

      # Accumulate performance gap percentage
      # average_percentage=$(echo "${average_percentage} + ${percentage}" | bc)
    done
    echo "" >> ${TEMP_REPORT}
    echo "" >> ${TEMP_CYCLES_REPORT}
    echo "" >> ${TEMP_INSTRET_REPORT}

  done
  echo "" >> ${TEMP_REPORT}
  echo "" >> ${TEMP_CYCLES_REPORT}
  echo "" >> ${TEMP_INSTRET_REPORT}
  # Average performance gap percentage
  # average_percentage=$(echo "scale=2; ${average_percentage}/${#SHAPE[@]} " | bc)

  # tabs=$(printf '\t%.0s' $(seq 1 ${#STAT_KEYWORD[@]}))
  # echo -e "average percentage\t${tabs}${average_percentage}" >> ${TEMP_REPORT}

  echo "" >> ${TEMP_REPORT}
  echo "" >> ${TEMP_REPORT}
  echo "" >> ${TEMP_CYCLES_REPORT}
  echo "" >> ${TEMP_CYCLES_REPORT}
  echo "" >> ${TEMP_INSTRET_REPORT}
  echo "" >> ${TEMP_INSTRET_REPORT}
done


echo "" >> ${TEMP_REPORT}
echo "" >> ${TEMP_REPORT}
echo "" >> ${TEMP_CYCLES_REPORT}
echo "" >> ${TEMP_CYCLES_REPORT}
echo "" >> ${TEMP_INSTRET_REPORT}
echo "" >> ${TEMP_INSTRET_REPORT}


# May add triton-cpu version?

echo "${GCC}" >> ${TEMP_REPORT}
${GCC} --version >> ${TEMP_REPORT}
echo "" >> ${TEMP_REPORT}

echo "${ZCC}" >> ${TEMP_REPORT}
${ZCC} --version >> ${TEMP_REPORT}
echo "" >> ${TEMP_REPORT}

echo "${GCC}" >> ${TEMP_CYCLES_REPORT}
${GCC} --version >> ${TEMP_CYCLES_REPORT}
echo "" >> ${TEMP_CYCLES_REPORT}

echo "${ZCC}" >> ${TEMP_CYCLES_REPORT}
${ZCC} --version >> ${TEMP_CYCLES_REPORT}
echo "" >> ${TEMP_CYCLES_REPORT}

echo "${GCC}" >> ${TEMP_INSTRET_REPORT}
${GCC} --version >> ${TEMP_INSTRET_REPORT}
echo "" >> ${TEMP_INSTRET_REPORT}

echo "${ZCC}" >> ${TEMP_INSTRET_REPORT}
${ZCC} --version >> ${TEMP_INSTRET_REPORT}
echo "" >> ${TEMP_INSTRET_REPORT}

# Align columns using column -t, but preserve header comments and version info
# Use column -t -s $'\t' to only split on tabs, not spaces
{
  # Process performance data sections with column -t
  awk '
    BEGIN { in_perf_section = 0; perf_lines = "" }
    /^##### .* kernel performance #####/ {
      if (perf_lines != "") {
        print perf_lines | "column -t -s \"\t\""
        close("column -t -s \"\t\"")
        perf_lines = ""
      }
      print
      in_perf_section = 1
      next
    }
    /^$/ && in_perf_section == 1 && perf_lines != "" {
      print perf_lines | "column -t -s \"\t\""
      close("column -t -s \"\t\"")
      perf_lines = ""
      in_perf_section = 0
      print
      next
    }
    in_perf_section == 1 {
      if (perf_lines != "") perf_lines = perf_lines "\n"
      perf_lines = perf_lines $0
      next
    }
    {
      print
    }
    END {
      if (perf_lines != "") {
        print perf_lines | "column -t -s \"\t\""
        close("column -t -s \"\t\"")
      }
    }
  ' ${TEMP_REPORT}
} > ${REPORT_FILE}

# Process cycles report
{
  awk '
    BEGIN { in_perf_section = 0; perf_lines = "" }
    /^##### .* kernel cycles #####/ {
      if (perf_lines != "") {
        print perf_lines | "column -t -s \"\t\""
        close("column -t -s \"\t\"")
        perf_lines = ""
      }
      print
      in_perf_section = 1
      next
    }
    /^$/ && in_perf_section == 1 && perf_lines != "" {
      print perf_lines | "column -t -s \"\t\""
      close("column -t -s \"\t\"")
      perf_lines = ""
      in_perf_section = 0
      print
      next
    }
    in_perf_section == 1 {
      if (perf_lines != "") perf_lines = perf_lines "\n"
      perf_lines = perf_lines $0
      next
    }
    {
      print
    }
    END {
      if (perf_lines != "") {
        print perf_lines | "column -t -s \"\t\""
        close("column -t -s \"\t\"")
      }
    }
  ' ${TEMP_CYCLES_REPORT}
} > ${CYCLES_REPORT}

# Process instret report
{
  awk '
    BEGIN { in_perf_section = 0; perf_lines = "" }
    /^##### .* kernel instret #####/ {
      if (perf_lines != "") {
        print perf_lines | "column -t -s \"\t\""
        close("column -t -s \"\t\"")
        perf_lines = ""
      }
      print
      in_perf_section = 1
      next
    }
    /^$/ && in_perf_section == 1 && perf_lines != "" {
      print perf_lines | "column -t -s \"\t\""
      close("column -t -s \"\t\"")
      perf_lines = ""
      in_perf_section = 0
      print
      next
    }
    in_perf_section == 1 {
      if (perf_lines != "") perf_lines = perf_lines "\n"
      perf_lines = perf_lines $0
      next
    }
    {
      print
    }
    END {
      if (perf_lines != "") {
        print perf_lines | "column -t -s \"\t\""
        close("column -t -s \"\t\"")
      }
    }
  ' ${TEMP_INSTRET_REPORT}
} > ${INSTRET_REPORT}

# Clean up temp file
rm -f ${TEMP_REPORT}
rm -f ${TEMP_CYCLES_REPORT}
rm -f ${TEMP_INSTRET_REPORT}
