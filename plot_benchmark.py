import pandas as pd

import matplotlib.pyplot as plt
import numpy as np

# 读取Excel文件
file_path = '/home/crux/workspace/AI-Kernel-Library/Benchmark/build/report.xls'  # 替换为你的Excel文件路径

#FIXME: Code refactor
# dropout_df = pd.read_csv(file_path, header=0, comment='#', usecols=['shape (NxRUN_COUNT)','gcc_T1',	'zcc_T1', 'triton_T1','gcc_T4',	'zcc_T4', 'triton_T4', 'gcc_T8',	'zcc_T8',	'triton_T8' ], skip_blank_lines=True, sep='\t', nrows=64)

# warp_data = pd.read_csv(file_path, header=0, comment='#', usecols=['shape (HxWxCxRUN_COUNT)','gcc_T1',	'zcc_T1', 'triton_T1','gcc_T4',	'zcc_T4', 'triton_T4', 'gcc_T8',	'zcc_T8',	'triton_T8' ], skip_blank_lines=True, sep='\t', nrows=98)
# warp_df = warp_data.iloc[::2]

# resize_data = pd.read_csv(file_path, header=0, comment='#', usecols=['shape (HxWxCxRUN_COUNT)','gcc_T1',	'zcc_T1', 'triton_T1','gcc_T4',	'zcc_T4', 'triton_T4', 'gcc_T8',	'zcc_T8',	'triton_T8' ], skip_blank_lines=True, sep='\t', nrows=98)
# resize_df = resize_data.iloc[::2]

# rope_data = pd.read_csv(file_path, header=0, comment='#', usecols=['shape (SEQ_LENxBATCH_NUMxHEAD_NUMxHEAD_DIMxRUN_COUNT)','gcc_T1',	'zcc_T1', 'triton_T1','gcc_T4',	'zcc_T4', 'triton_T4', 'gcc_T8',	'zcc_T8',	'triton_T8' ], skip_blank_lines=True, sep='\t', nrows=90)
# rope_df = rope_data.iloc[::2]

# correlation_data = pd.read_csv(file_path, header=0, comment='#', usecols=['shape (OUT_CHANNELxIN_CHANNELxHEIGHTxWIDTHxRUN_COUNT)','gcc_T1',	'zcc_T1', 'triton_T1','gcc_T4',	'zcc_T4', 'triton_T4', 'gcc_T8',	'zcc_T8',	'triton_T8'
# ], skip_blank_lines=True, sep='\t', nrows=96)
# correlation_df = correlation_data.iloc[::2]

# layernorm_data = pd.read_csv(file_path, header=0, comment='#',  usecols=['shape (NxDxRUN_COUNT)','gcc_T1',	'zcc_T1', 'triton_T1','gcc_T4',	'zcc_T4', 'triton_T4', 'gcc_T8',	'zcc_T8',	'triton_T8'], skip_blank_lines=True, sep='\t', nrows=128)
# # two many data, filter
# layernorm_df = layernorm_data.iloc[::2]


# matmul_data = pd.read_csv(file_path, header=0, comment='#',  usecols=['shape (MxNxKxRUN_COUNT)','gcc_T1',	'zcc_T1', 'triton_T1','gcc_T4',	'zcc_T4', 'triton_T4', 'gcc_T8',	'zcc_T8',	'triton_T8'], skip_blank_lines=True, sep='\t', nrows=128)
# matmul_df = matmul_data.iloc[::2]

# softmax_data = pd.read_csv(file_path, header=0, comment='#',  usecols=['shape (RxCxRUN_COUNT)','gcc_T1',	'zcc_T1', 'triton_T1','gcc_T4',	'zcc_T4', 'triton_T4',  'gcc_T8',	'zcc_T8',	'triton_T8'], skip_blank_lines=True, sep='\t', nrows=91)
# # two many data, filter
# softmax_df = softmax_data.iloc[::2]


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
    result = value  # 如果没有找到'x'，就不做任何修改
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

      # Make thread 4 data transparent to avoid overlapping with thread 8 data
      if((i + 2) // modulo == 2):
        plt.plot(x_processed, y_processed, marker=marks[i % modulo],color='gray', alpha=0.5, linewidth=0.5, linestyle=linestyle_str[i % modulo], label=data_frame.columns[i])
      else:
        plt.plot(x_processed, y_processed, marker=marks[i % modulo], linestyle=linestyle_str[i % modulo], label=data_frame.columns[i])

  # Customizing the plot
  plt.title(kernel_name + " kernel performance")
  plt.xlabel(kernel_shape)

  plt.ylabel('Running time: GB/s')
  # plt.yscale('log')  # 使用对数刻度
  # plt.ylim(0.02, 0.06)  # 设置纵轴范围

  # 将X轴标签移动到末端（坐标系中的 1, -0.1）
  ax.xaxis.set_label_coords(1.0, 0.01)

  # 将Y轴标签移动到末端（坐标系中的 -0.1, 1）
  ax.yaxis.set_label_coords(0, 1.0)
  ax.yaxis.label.set_rotation(0)


  plt.legend(loc='upper right', fontsize='small')
  plt.legend(fontsize=12)  # 你可以根据需要调整数值
  plt.xticks(rotation=90)
  # plt.grid(True)
  # Show the plot
  plt.show()

#FIXME: Code refactor
# plot(correlation_df, "correlation", "OUT_CHANNEL x IN_CHANNEL x HEIGHT x WIDTH")
# plot(layernorm_df, "layernorm", "N x D")
# plot(matmul_df, "matmul", "M x N x K")
# plot(softmax_df, "softmax", "R x C")
# plot(dropout_df, "dropout", "N")

# plot(warp_df, "warp", "H x W x C")

# plot(resize_df, "resize", "H x W x C")
# plot(rope_df, "rope", "SEQ_LEN x BATCH_NUM x HEAD_NUM x HEAD_DIM")
