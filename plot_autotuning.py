import pandas as pd

import matplotlib.pyplot as plt
import numpy as np

#############################################################################
######        Only supports tuning one kernel once a time             #######
######        Need config report file path                            #######
#############################################################################

# Define the list of plot configurations
plot_configs = [
    {"kernel_name": "correlation", "tuning_param": "BLOCK_SIZE", "report_file":"/home/crux/workspace/AI-Kernel-Library/Benchmark/build/report.xls"},
    {"kernel_name": "layernorm", "tuning_param": "BLOCK_SIZE", "report_file":"/home/crux/workspace/AI-Kernel-Library/Benchmark/build/report.xls"},
    {"kernel_name": "softmax", "tuning_param": "BLOCK_SIZE", "report_file":"/home/crux/workspace/AI-Kernel-Library/Benchmark/build/report.xls"},
    {"kernel_name": "matmul", "tuning_param": "BLOCK_SIZE", "report_file":"/home/crux/workspace/AI-Kernel-Library/Benchmark/build/report.xls"},
    {"kernel_name": "resize", "tuning_param": "BLOCK_SIZE_H x BLOCK_SIZE_W", "report_file":"/home/crux/workspace/AI-Kernel-Library/Benchmark/build/report.xls"},
]


# | #####  softmax_kernel kernel performance ##### |
# | shape (RxCxRUN_COUNT) | gcc_T1  | zcc_T1  | triton_T1_softmax_kernel_8 | triton_T1_softmax_kernel_64 | gcc_T4 | zcc_T4  | triton_T4_softmax_kernel_8 | triton_T4_softmax_kernel_64 | ...  |
# | --------------------- | ------- | ------- | -------------------------- | --------------------------- | ------ | ------- | -------------------------- | --------------------------- | -T8- |
# | 1823x781x100          | 7.12539 | 6.93295 | 7.07778                    | 6.90874                     | 1.8199 | 1.82919 | 1.81947                    | 1.76622                     | ...  |

def split_block_shape(value):
  # Splitting a string
  parts = value.split('_')

  # FIXME: Check if there are enough parts and get the string after the second underscore
  if len(parts) > 2:
      # Concat the part after the second underscore
      result = '_'.join(parts[2:])
  else:
      # If there is no second underscore, return an empty string
      result = ''
  return result

def plot(data_frame, kernel_name, block_shape):
  # print(kernel_name, " data\n", data_frame)
  # print(data_frame.shape)

  # Customizing the plot
  # Prepare plot
  fig, ax = plt.subplots(figsize=(14, 20))

  total_tuning_shape=(data_frame.shape[1]-1)//3
  # print("total_tuning_shape\n", total_tuning_shape)

  marks=["o", "*", "s", "^"]
  linestyle_str = [
    'solid',
    'dotted',
    'dashed',
    'dashdot',
  ]
  for i in [1, 4, 8]:
    triton_start=3 + total_tuning_shape * (i//4)
    triton_end=total_tuning_shape * (i//4 + 1) + 1
    # print(triton_start,triton_end)

    x_processed = [ split_block_shape(data_frame.columns[i]) for i in range(triton_start,triton_end)]
    # print(x_processed)

    gcc=data_frame.iloc[0:data_frame.shape[0], 1 + total_tuning_shape * (i//4)]
    plt.plot(x_processed, [gcc[0]]* (total_tuning_shape-2), marker=marks[1], linestyle=linestyle_str[1], label=data_frame.columns[1 + total_tuning_shape * (i//4)])
    # print(gcc)

    zcc=data_frame.iloc[0:data_frame.shape[0], 2 + total_tuning_shape * (i//4)]
    # print(zcc)
    plt.plot(x_processed, [zcc[0]] * (total_tuning_shape-2), marker=marks[2], linestyle=linestyle_str[2], label=data_frame.columns[2 + total_tuning_shape * (i//4)])



    triton = data_frame.iloc[0:data_frame.shape[0], triton_start:triton_end]
    plt.plot(x_processed, triton.iloc[0], marker=marks[3], linestyle=linestyle_str[3],  label="triton_T"+str(i))


    # print("gcc_T"+str(i),"\n", gcc)
    # print("zcc_T"+str(i),"\n", zcc)
    # print("x_processed\n", x_processed)
    # print("triton_T"+str(i),"\n", triton)

  # Customizing the plot
  plt.title(kernel_name + " kernel performance")
  plt.xlabel(block_shape)

  plt.ylabel('Running time: s')

  # Move the x-axis label to the end (1, -0.1 in the coordinate system)
  ax.xaxis.set_label_coords(1.0, 0.01)

  # Move the Y axis label to the end (-0.1, 1 in the coordinate system)
  ax.yaxis.set_label_coords(0, 1.0)
  ax.yaxis.label.set_rotation(0)


  plt.legend(loc='upper right', fontsize='small')
  plt.xticks(rotation=90)
  # plt.grid(True)
  # Show the plot
  plt.show()

# Iterate over each configuration and call the plot function
for config in plot_configs:
    file_path = config["report_file"]
    data_df = pd.read_csv(file_path, header=0, comment='#', skip_blank_lines=True, sep='\t', nrows=1)
    plot(data_df, config["kernel_name"], config["tuning_param"])