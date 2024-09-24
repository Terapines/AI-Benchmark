import pandas as pd

import matplotlib.pyplot as plt
import numpy as np

# 读取Excel文件
file_path = '/home/crux/workspace/AI-Kernel-Library/Benchmark/build/report.xls'  # 替换为你的Excel文件路径

# correlation_df = pd.read_csv(file_path, header=0, comment='#', skip_blank_lines=True, sep='\t', nrows=1)
# layernorm_df = pd.read_csv(file_path, header=0, comment='#', skip_blank_lines=True, sep='\t', nrows=1)
# matmul_df = pd.read_csv(file_path, header=0, comment='#', skip_blank_lines=True, sep='\t', nrows=1)
# softmax_df = pd.read_csv(file_path, header=497 - 4 - 1, comment='#', skip_blank_lines=True, sep='\t', nrows=1)
resize_df = pd.read_csv(file_path, header=0, comment='#', skip_blank_lines=True, sep='\t', nrows=1)


def split_block_shape(value):
  # 使用 split 方法分割字符串
  parts = value.split('_')

  # 检查是否有足够的部分，并获取第二个下划线后的字符串
  if len(parts) > 2:
      result = '_'.join(parts[2:])  # 将第二个下划线后面的部分连接起来
  else:
      result = ''  # 如果没有第二个下划线，返回空字符串
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
    'solid', # Same as (0, ()) or '-'；solid’， (0, ()) ， '-'三种都代表实线。
    'dotted',  # Same as (0, (1, 1)) or '.'
    'dashed',  # Same as '--'
    'dashdot',  # Same as '-.'
  ]
  for i in [1, 4, 8]:
    triton_start=3 + total_tuning_shape * (i//4)
    triton_end=total_tuning_shape * (i//4 + 1)
    x_processed = [ split_block_shape(data_frame.columns[i]) for i in range(triton_start,triton_end)]

    gcc=data_frame.iloc[0:data_frame.shape[0], 1 + total_tuning_shape * (i//4)]
    plt.plot(x_processed, [gcc[0]]* (total_tuning_shape-3), marker=marks[1], linestyle=linestyle_str[1], label=data_frame.columns[1 + total_tuning_shape * (i//4)])

    zcc=data_frame.iloc[0:data_frame.shape[0], 2 + total_tuning_shape * (i//4)]
    plt.plot(x_processed, [zcc[0]] * (total_tuning_shape-3), marker=marks[2], linestyle=linestyle_str[2], label=data_frame.columns[2 + total_tuning_shape * (i//4)])



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

# plot(correlation_df, "correlation", "BLOCK_SIZE")
# plot(layernorm_df, "layernorm", "BLOCK_SIZE")
# plot(softmax_df, "softmax", "BLOCK_SIZE")
# plot(matmul_df, "matmul", "BLOCK_SIZE")
plot(resize_df, "resize", "BLOCK_SIZE_H x BLOCK_SIZE_W")