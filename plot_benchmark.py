import pandas as pd

import matplotlib.pyplot as plt
import numpy as np

# 读取Excel文件
file_path = '/home/crux/workspace/AI-Kernel-Library/Benchmark/bench-multi-shape/report.xls'  # 替换为你的Excel文件路径


correlation_df = pd.read_csv(file_path, header=0, comment='#', usecols=['shape (OUT_CHANNELxIN_CHANNELxHEIGHTxWIDTHxRUN_COUNT)','gcc_T1',	'zcc_T1', 'triton_T1', 'gcc_T8',	'zcc_T8',	'triton_T8'
], skip_blank_lines=True, sep='\t', nrows=96)


layernorm_data = pd.read_csv(file_path, header=103 - 2 - 1, comment='#',  usecols=['shape (NxDxRUN_COUNT)','gcc_T1',	'zcc_T1', 'triton_T1', 'triton-v2_T1', 'gcc_T8',	'zcc_T8',	'triton_T8', 'triton-v2_T8',
], skip_blank_lines=True, sep='\t', nrows=256)
# two many data, filter
layernorm_df = layernorm_data.iloc[::2]


matmul_df = pd.read_csv(file_path, header=364 - 3 - 1, comment='#',  usecols=['shape (MxNxKxRUN_COUNT)','gcc_T1',	'zcc_T1', 'triton_T1',  'gcc_T8',	'zcc_T8',	'triton_T8',
], skip_blank_lines=True, sep='\t', nrows=128)


softmax_data = pd.read_csv(file_path, header=497 - 4 - 1, comment='#',  usecols=['shape (RxCxRUN_COUNT)','gcc_T1',	'zcc_T1', 'triton_T1',  'gcc_T8',	'zcc_T8',	'triton_T8',
], skip_blank_lines=True, sep='\t', nrows=256)
# two many data, filter
softmax_df = softmax_data.iloc[::2]


def norm_performance(shape, value):
    # Convert to string and split by 'x'
    parts = str(shape).split('x')
    # Convert parts to float and multiply them
    multiplied = np.prod([float(part) for part in parts if part])
    # Multiply the result with the original value
    return multiplied/value/10e9

def split_runcount(value):
  last_x_index = value.rfind('x')

  if last_x_index != -1:
    result = value[:last_x_index]
  else:
    result = data  # 如果没有找到'x'，就不做任何修改
  return result


def plot(data_frame, kernel_name, kernel_shape, modulo=3):
  print(kernel_name, " data\n", data_frame)
  print(data_frame.shape)


  # Customizing the plot
  # Prepare plot
  fig, ax = plt.subplots(figsize=(14, 20))

  marks=["o", "*", "s", "^"]
  linestyle_str = [
    'solid', # Same as (0, ()) or '-'；solid’， (0, ()) ， '-'三种都代表实线。
    'dotted',  # Same as (0, (1, 1)) or '.'
    'dashed',  # Same as '--'
    'dashdot',  # Same as '-.'
  ]
  for i in range(1, data_frame.shape[1]):
      x = data_frame.iloc[0:data_frame.shape[0], 0]
      y_original = data_frame.iloc[0:data_frame.shape[0], i]

      x_processed = [ split_runcount(value) for value in x]
      y_processed = [ norm_performance(x_value,y_value) for x_value, y_value in zip(x,y_original)]

      plt.plot(x_processed, y_processed, marker=marks[i % modulo], linestyle=linestyle_str[i % modulo], label=data_frame.columns[i])

  # Customizing the plot
  plt.title(kernel_name + " kernel performance")
  plt.xlabel(kernel_shape)

  plt.ylabel('Running time: GB/s')

  # 将X轴标签移动到末端（坐标系中的 1, -0.1）
  ax.xaxis.set_label_coords(1.0, 0.01)

  # 将Y轴标签移动到末端（坐标系中的 -0.1, 1）
  ax.yaxis.set_label_coords(0, 1.0)
  ax.yaxis.label.set_rotation(0)


  plt.legend(loc='upper right', fontsize='small')
  plt.xticks(rotation=90)
  # plt.grid(True)
  # Show the plot
  plt.show()


plot(correlation_df, "correlation", "OUT_CHANNEL x IN_CHANNEL x HEIGHT x WIDTH")
plot(layernorm_df, "layernorm", "N x D",4)
plot(matmul_df, "matmul", "M x N x K x RUN_COUNT")
plot(softmax_df, "softmax", "R x C x RUN_COUNT")
