import json
import matplotlib.pyplot as plt
import os
import glob

# --- Configuration ---
# Define the path to your experiment's working directory
work_dir = './segmentation/work_dirs/exp_v5_aug/'
output_filename = 'loss_curve.png'
# --- End of Configuration ---

# Find the log file automatically
try:
    log_file_path = glob.glob(os.path.join(work_dir, '2025*.log.json'))[0]
    print(f"Found log file: {log_file_path}")
except IndexError:
    print(f"Error: Could not find a .log.json file starting with '2025' in {work_dir}")
    exit()

# Lists to store the data for plotting
iterations = []
losses = []

# Read the log file line by line
with open(log_file_path, 'r') as f:
    for line in f:
        try:
            log_data = json.loads(line.strip())
            # We only care about training data, which has a 'loss' key
            if 'loss' in log_data and 'iter' in log_data:
                iterations.append(log_data['iter'])
                losses.append(log_data['loss'])
        except json.JSONDecodeError:
            # Ignore lines that are not valid JSON
            continue

if not iterations:
    print("Error: No training loss data found in the log file.")
    exit()

# Create the plot
plt.figure(figsize=(12, 6))
plt.plot(iterations, losses, label='Training Loss')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.title('Training Loss Curve')
plt.legend()
plt.grid(True)

# Save the plot to a file
plt.savefig(output_filename)

print(f"\nSuccessfully generated the plot and saved it as '{output_filename}'")
print(f"Total iterations plotted: {len(iterations)}")
