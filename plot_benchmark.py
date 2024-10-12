import pandas as pd

import matplotlib.pyplot as plt
import numpy as np

#############################################################################
######        Only supports plot one kernel once a time               #######
######        Need config report file path                            #######
#############################################################################

# Define configurations for each dataset
datasets = [
    {
        'name': 'warp',
        'report_file': '/home/crux/workspace/AI-Kernel-Library/Benchmark/build/report.xls',
        'usecols': [
            'shape (HxWxCxRUN_COUNT)', 'gcc_T1', 'zcc_T1', 'triton_T1',
            'gcc_T4', 'zcc_T4', 'triton_T4',
            'gcc_T8', 'zcc_T8', 'triton_T8'
        ],
        'nrows': 98,
        'shape_label': 'H x W x C',
        'xaxis': (0.96, 0.02),
    },
    {
        'name': 'resize',
        'report_file': '/home/crux/workspace/AI-Kernel-Library/Benchmark/build/report.xls',
        'usecols': [
            'shape (HxWxCxRUN_COUNT)', 'gcc_T1', 'zcc_T1', 'triton_T1',
            'gcc_T4', 'zcc_T4', 'triton_T4',
            'gcc_T8', 'zcc_T8', 'triton_T8'
        ],
        'nrows': 98,
        'shape_label': 'H x W x C',
        'xaxis': (0.96, 0.02)
    },
    {
        'name': 'rope',
        'report_file': '/home/crux/workspace/AI-Kernel-Library/Benchmark/build/report.xls',
        'usecols': [
            'shape (SEQ_LENxBATCH_NUMxHEAD_NUMxHEAD_DIMxRUN_COUNT)', 'gcc_T1', 'zcc_T1', 'triton_T1',
            'gcc_T4', 'zcc_T4', 'triton_T4',
            'gcc_T8', 'zcc_T8', 'triton_T8'
        ],
        'nrows': 89,
        'shape_label': 'SEQ_LEN x BATCH_NUM x HEAD_NUM x HEAD_DIM',
        'xaxis': (0.82, 0.02)
    },
    {
        'name': 'correlation',
        'report_file': '/home/crux/workspace/AI-Kernel-Library/Benchmark/build/report.xls',
        'usecols': [
            'shape (OUT_CHANNELxIN_CHANNELxHEIGHTxWIDTHxRUN_COUNT)', 'gcc_T1', 'zcc_T1', 'triton_T1',
            'gcc_T4', 'zcc_T4', 'triton_T4',
            'gcc_T8', 'zcc_T8', 'triton_T8'
        ],
        'nrows': 96,
        'shape_label': 'OUT_CHANNEL x IN_CHANNEL x HEIGHT x WIDTH',
        'xaxis': (0.84, 0.02)
    },
    {
        'name': 'layernorm',
        'report_file': '/home/crux/workspace/AI-Kernel-Library/Benchmark/build/report.xls',
        'usecols': [
            'shape (NxDxRUN_COUNT)', 'gcc_T1', 'zcc_T1', 'triton_T1',
            'gcc_T4', 'zcc_T4', 'triton_T4',
            'gcc_T8', 'zcc_T8', 'triton_T8'
        ],
        'nrows': 128,
        'shape_label': 'N x D',
        'xaxis': (0.98, 0.02)
    },
    {
        'name': 'matmul',
        'report_file': '/home/crux/workspace/AI-Kernel-Library/Benchmark/build/report.xls',
        'usecols': [
            'shape (MxNxKxRUN_COUNT)', 'gcc_T1', 'zcc_T1', 'triton_T1',
            'gcc_T4', 'zcc_T4', 'triton_T4',
            'gcc_T8', 'zcc_T8', 'triton_T8'
        ],
        'nrows': 128,
        'shape_label': 'M x N x K',
        'xaxis': (0.96, 0.02)
    },
    {
        'name': 'softmax',
        'report_file': '/home/crux/workspace/AI-Kernel-Library/Benchmark/build/report.xls',
        'usecols': [
            'shape (RxCxRUN_COUNT)', 'gcc_T1', 'zcc_T1', 'triton_T1',
            'gcc_T4', 'zcc_T4', 'triton_T4',
            'gcc_T8', 'zcc_T8', 'triton_T8'
        ],
        'nrows': 91,
        'shape_label': 'R x C',
        'xaxis': (0.98, 0.02)
    }
]

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
    result = value
  return result


def plot(data_frame, kernel_name, kernel_shape, kernel_xaixs, modulo=3):
  print(kernel_name, " data\n", data_frame)
  print(data_frame.shape)


  # Customizing the plot
  # Prepare plot
  fig, ax = plt.subplots(figsize=(14, 20))

  marks=["o", "*", "s", "^"]
  linestyle_str = [
    'solid',
    'dotted',
    'dashed',
    'dashdot',
  ]
  thread_1 = []
  acc_4_to_1 = []
  acc_8_to_1 = []
  for i in range(1, data_frame.shape[1]):
      x = data_frame.iloc[0:data_frame.shape[0], 0]
      y_original = data_frame.iloc[0:data_frame.shape[0], i]

      x_processed = [ split_runcount(value) for value in x]
      y_processed = [ norm_performance(x_value,y_value) for x_value, y_value in zip(x,y_original)]

      # if ((i + 2) // modulo == 1):
      #   thread_1 = y_processed
      # elif((i + 2) // modulo == 2):
      #   acc_4_to_1 = [ T4 / T1 for T4, T1 in zip(y_processed, thread_1)]
      #   print("ave acc_4_to_1 : ", sum(acc_4_to_1)/len(acc_4_to_1))
      #   print("min acc_4_to_1 : ", min(acc_4_to_1))
      #   print("max acc_4_to_1 : ", max(acc_4_to_1))
      # else :
      #   acc_8_to_1 = [ T4 / T1 for T4, T1 in zip(y_processed, thread_1)]
      #   print("acc_8_to_1 : ", sum(acc_8_to_1)/len(acc_8_to_1))
      #   print("min acc_8_to_1 : ", min(acc_8_to_1))
      #   print("max acc_8_to_1 : ", max(acc_8_to_1))
      print(sum(y_processed)/len(y_processed))

      # Make thread 4 data transparent to avoid overlapping with thread 8 data
      if((i + 2) // modulo == 2):
        plt.plot(x_processed, y_processed, marker=marks[i % modulo],color='gray', alpha=0.5, linewidth=0.5, linestyle=linestyle_str[i % modulo], label=data_frame.columns[i])
      else:
        plt.plot(x_processed, y_processed, marker=marks[i % modulo], linestyle=linestyle_str[i % modulo], label=data_frame.columns[i])

  # Customizing the plot
  plt.title(kernel_name + " kernel performance")
  plt.xlabel(kernel_shape)

  plt.ylabel('GB/s')
  # Use logarithmic scale
  # plt.yscale('log')
  # Set the vertical axis range
  # plt.ylim(0.02, 0.06)

  # Move the x-axis label to the end
  ax.xaxis.set_label_coords(kernel_xaixs[0],kernel_xaixs[1])

  # Move the Y axis label to the end (-0.1, 1 in the coordinate system)
  ax.yaxis.set_label_coords(0.02, 1.0)
  ax.yaxis.label.set_rotation(0)


  plt.legend(loc='upper right', fontsize='small')
  plt.legend(fontsize=12)
  plt.xticks(rotation=90)
  # plt.grid(True)
  # Show the plot
  plt.show()

# Iterate through each dataset configuration
for dataset in datasets:
    # Read and process data
    data = pd.read_csv(
        dataset['report_file'],
        header=0,
        comment='#',
        usecols=dataset['usecols'],
        skip_blank_lines=True,
        sep='\t',
        nrows=dataset['nrows']
    )

    # Apply any data processing steps if necessary
    df = data.iloc[::2]  # Selecting every other row

    # Plot data
    plot(
        df,
        dataset['name'],
        dataset['shape_label'],
        dataset.get('xaxis')
    )