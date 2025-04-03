import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


# Define the MSE functions
def coarse_grain(series, scale):
    n = len(series)
    num_windows = n // scale
    coarse_grained_series = [series[i * scale:(i + 1) * scale].mean() for i in range(num_windows)]
    return np.array(coarse_grained_series)

def sample_entropy(time_series, m, r):
    N = len(time_series)
    B = 0.0
    A = 0.0
    xmi = np.array([time_series[i:i + m] for i in range(N - m)])
    xm1i = np.array([time_series[i:i + m + 1] for i in range(N - m - 1)])
    for i in range(len(xmi)):
        distance = np.abs(xmi[i] - xmi).sum(axis=1)
        B += np.sum(distance <= r)
    for i in range(len(xm1i)):
        distance = np.abs(xm1i[i] - xm1i).sum(axis=1)
        A += np.sum(distance <= r)
    B -= N - m
    A -= N - m - 1
    return -np.log(A / B)

def multiscale_entropy(time_series, max_scale, m, r):
    entropies = []
    for scale in range(1, max_scale + 1):
        cg_series = coarse_grain(time_series, scale)
        se = sample_entropy(cg_series, m, r)
        entropies.append(se)
    return entropies


# Directory where your files are located
data_dir = r'C:\Users\se006\OneDrive\Documents\Sahar\PhD\MDPI\gazebasevr\data'
output_dir = r'C:\Users\se006\OneDrive\Documents\Sahar\PhD\MDPI\Codes\Output\Y'

# Initialize a dictionary to store all MSE results
all_mse_results = {}

# Get the list of all CSV files
file_names = [f for f in os.listdir(data_dir) if f.endswith('.csv')]

# Process each file with a progress bar
for file_name in tqdm(file_names, desc='Processing files', unit='file'):
    file_path = os.path.join(data_dir, file_name)
    dataset = pd.read_csv(file_path)
    
    # Assuming 'x' column for analysis
    time_series_data = dataset['y'].dropna()

    # Define parameters for MSE calculation
    max_scale = 20
    m = 2  # Embedding dimension
    r = 0.2 * np.std(time_series_data)  # Distance threshold
    
    # Compute the Multiscale Entropy
    mse_results = multiscale_entropy(time_series_data, max_scale, m, r)
    
    # Store the results
    all_mse_results[file_name] = mse_results

# Save the raw MSE data to a CSV file
mse_df = pd.DataFrame.from_dict(all_mse_results, orient='index').transpose()
mse_df.to_csv(os.path.join(output_dir, 'MSE_results.csv'), index_label='Scale Factor')

# Function to select a subset of files for plotting
def select_files_for_plotting(file_dict, num_plots=6):
    average_mse_per_file = {file_name: np.mean(mse_values) for file_name, mse_values in file_dict.items()}
    sorted_files = sorted(average_mse_per_file, key=average_mse_per_file.get)
    selected_files = [sorted_files[0], sorted_files[-1]]
    interval = len(sorted_files) // (num_plots - 1)
    selected_files += [sorted_files[i * interval] for i in range(1, num_plots - 1)]
    return selected_files

# Select files to plot based on variability
selected_file_names = select_files_for_plotting(all_mse_results)

# Plot and save the MSE results for the selected files
for i, file_name in enumerate(selected_file_names, 1):
    plt.figure(figsize=(10, 6))
    scales = range(1, len(all_mse_results[file_name]) + 1)
    plt.plot(scales, all_mse_results[file_name], marker='o', label=file_name)
    plt.title(f'Multiscale Entropy Analysis of {file_name}')
    plt.xlabel('Scale Factor')
    plt.ylabel('Sample Entropy')
    plt.legend()
    plt.grid(True)
    # Save each individual plot
    plt.savefig(os.path.join(output_dir, f'MSE_plot_{i}.png'))
    plt.clf()
