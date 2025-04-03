import pandas as pd
import matplotlib.pyplot as plt

# Load the MSE results from the CSV file
mse_results_path = r"C:\Users\se006\OneDrive\Documents\Sahar\PhD\MDPI\Codes\Output\Y\MSE_results.csv"
mse_df = pd.read_csv(mse_results_path, index_col='Scale Factor')

# Define the task types based on your data
task_types = ['VRG', 'PUR', 'VID', 'TEX', 'RAN']

# Create a plot for each task type
for task in task_types:
    task_df = mse_df.filter(regex=fr'_{task}\.csv$')  # Filter columns for the task type
    
    plt.figure(figsize=(15, 8))
    
    for column in task_df.columns:
        plt.plot(task_df.index, task_df[column], label=column, marker='o')
    
    # Set bold and enlarged labels for the axes and title
    title_text = f'(b): MES for task type: {task} - Vertical Gaze Position (column y)'
    plt.title(title_text, fontsize=16, fontweight='bold')
    plt.xlabel('Scale Factor', fontsize=14, fontweight='bold')
    plt.ylabel('Sample Entropy', fontsize=14, fontweight='bold')
    
    # Enlarge and bold the axis tick labels
    plt.xticks(range(0, 21, 2), fontsize=12, fontweight='bold')
    plt.yticks(fontsize=12, fontweight='bold')

    # Remove the grid and set the legend with bold font
    plt.grid(False)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=4, fontsize=10, title_fontsize='large', frameon=False, prop={'weight':'bold'})

    # Save the figure
    plt.savefig(f"C:/Users/se006/OneDrive/Documents/Sahar/PhD/MDPI/Codes/Output/Y/b_MSE_{task}.png", bbox_inches='tight')
    plt.close()
