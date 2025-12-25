import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re

# Read the report file
file_path = '/home/xinyi/workspace/AI-Benchmark/build/report.xls'

with open(file_path, 'r') as f:
    lines = f.readlines()
    header_line = lines[1].strip()  # Line 2
    data_line = lines[2].strip()    # Line 3

# Parse header and data
headers = header_line.split('\t')
data = data_line.split('\t')

shape = data[0]
kernel_names = headers[1:]
running_times = [float(x) for x in data[1:]]

# Extract BLOCKSIZE from kernel name and group by T1, T4, T8
def extract_block_size(kernel_name):
    # Extract the last three numbers from kernel name (e.g., "16_16_16" from "triton_T1_matmul_kernel_16_16_16")
    match = re.search(r'(\d+)_(\d+)_(\d+)$', kernel_name)
    if match:
        return tuple(map(int, match.groups()))  # Return as tuple for sorting
    return None

groups = {'T1': [], 'T4': [], 'T8': []}
for name, time in zip(kernel_names, running_times):
    block_size = extract_block_size(name)
    if block_size is None:
        continue

    if '_T1_' in name:
        groups['T1'].append((block_size, time, name))
    elif '_T4_' in name:
        groups['T4'].append((block_size, time, name))
    elif '_T8_' in name:
        groups['T8'].append((block_size, time, name))

# Sort each group by BLOCKSIZE
for group_name in groups:
    groups[group_name].sort(key=lambda x: x[0])

# Create a single plot with three lines
fig, ax = plt.subplots(figsize=(16, 8))

colors = {'T1': 'blue', 'T4': 'green', 'T8': 'red'}
markers = {'T1': 'o', 'T4': 's', 'T8': '^'}
linestyles = {'T1': '-', 'T4': '--', 'T8': '-.'}

for group_name in ['T1', 'T4', 'T8']:
    if not groups[group_name]:
        continue

    block_sizes = [f"{x[0][0]}_{x[0][1]}_{x[0][2]}" for x in groups[group_name]]
    times = [x[1] for x in groups[group_name]]

    # Plot line with markers
    ax.plot(block_sizes, times,
            marker=markers[group_name],
            linestyle=linestyles[group_name],
            color=colors[group_name],
            linewidth=2,
            markersize=8,
            label=f'{group_name}',
            alpha=0.8)

# Customize the plot
ax.set_title(f'Kernel Performance Comparison for Shape: {shape}', fontsize=14, fontweight='bold')
ax.set_xlabel('BLOCKSIZE (MxNxK)', fontsize=12)
ax.set_ylabel('Running Time (s)', fontsize=12)
ax.set_xticks(block_sizes)  # Use T1's block sizes as reference (they should all have the same)
ax.set_xticklabels(block_sizes, rotation=45, ha='right', fontsize=9)
ax.grid(True, alpha=0.3, linestyle='--')
ax.legend(loc='best', fontsize=11)

# Find and mark the minimum time for each group
for group_name in ['T1', 'T4', 'T8']:
    if not groups[group_name]:
        continue
    min_idx = min(range(len(groups[group_name])), key=lambda i: groups[group_name][i][1])
    min_block_size = f"{groups[group_name][min_idx][0][0]}_{groups[group_name][min_idx][0][1]}_{groups[group_name][min_idx][0][2]}"
    min_time = groups[group_name][min_idx][1]
    ax.plot(min_block_size, min_time,
            marker='*', markersize=15, color=colors[group_name],
            markeredgecolor='black', markeredgewidth=1,
            label=f'{group_name} min' if group_name == 'T1' else '')

# Update legend to include min markers
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, labels, loc='best', fontsize=11)

plt.tight_layout()
plt.savefig('/home/xinyi/workspace/AI-Benchmark/build/report_plot_lines.png', dpi=300, bbox_inches='tight')
print(f"Plot saved to /home/xinyi/workspace/AI-Benchmark/build/report_plot_lines.png")
plt.show()
