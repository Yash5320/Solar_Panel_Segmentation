import json
import matplotlib.pyplot as plt
import os
import glob
import numpy as np

# --- Configuration ---
work_dir = './segmentation/work_dirs/exp_v5_aug/'
output_filename = 'training_and_validation_curves.png'
# --- End of Configuration ---

try:
    log_file_path = glob.glob(os.path.join(work_dir, '2025*.log.json'))[0]
    print(f"Found log file: {log_file_path}")
except IndexError:
    print(f"Error: Could not find a .log.json file in {work_dir}")
    exit()

train_iters, train_losses = [], []
val_iters, val_miou = [], []

with open(log_file_path, 'r') as f:
    for line in f:
        try:
            log_data = json.loads(line.strip())
            # Capture training loss
            if 'loss' in log_data and 'iter' in log_data and log_data.get('mode') == 'train':
                train_iters.append(log_data['iter'])
                train_losses.append(log_data['loss'])
            # Capture validation mIoU
            if 'mIoU' in log_data and 'iter' in log_data and log_data.get('mode') == 'val':
                val_iters.append(log_data['iter'])
                val_miou.append(log_data['mIoU'])
        except json.JSONDecodeError:
            continue

if not train_iters:
    print("Error: No training data found.")
    exit()

# Create the plot with two y-axes
fig, ax1 = plt.subplots(figsize=(12, 6))

# Plot Training Loss
ax1.set_xlabel('Iterations')
ax1.set_ylabel('Training Loss', color='tab:blue')
ax1.plot(train_iters, train_losses, color='tab:blue', label='Training Loss')
ax1.tick_params(axis='y', labelcolor='tab:blue')
ax1.grid(True)

# Create a second y-axis for Validation mIoU if data exists
if val_iters:
    ax2 = ax1.twinx()
    ax2.set_ylabel('Validation mIoU (%)', color='tab:red')
    ax2.plot(val_iters, val_miou, color='tab:red', marker='o', linestyle='--', label='Validation mIoU')
    ax2.tick_params(axis='y', labelcolor='tab:red')

fig.tight_layout()
plt.title('Training Loss and Validation mIoU Curves')
plt.savefig(output_filename)
print(f"\nSuccessfully generated plot: '{output_filename}'")
