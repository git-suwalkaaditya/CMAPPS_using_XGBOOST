import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load your dataset (update the path to your CSV file location)
df = pd.read_csv("C:/Users\hp\Downloads\CMAPSSData/train_FD001.csv")

# Define sensor columns (based on your dataset)
sensors = [f'sensor_{i}' for i in range(1, 22)]  # Assuming there are 21 sensors

# List of engines to process (1 to 5)
engines = [1, 2, 3, 4, 5]

# Loop through the first 5 engines
for engine_id in engines:
    # Filter data for the specific engine
    df_engine = df[df['engine_id'] == engine_id]

    # Normalize the sensor data using z-score normalization for each engine separately
    for sensor in sensors:
        # Calculate mean and standard deviation for each sensor for this engine
        mu = df_engine[sensor].mean()
        sigma = df_engine[sensor].std()

        # Apply the normalization formula: 2 * (x - mean) / std - 1
        df_engine[f'{sensor}_norm'] = 2 * (df_engine[sensor] - mu) / sigma - 1

    # Create a grid of subplots (7 rows and 3 columns for 21 sensors)
    fig, axes = plt.subplots(nrows=7, ncols=3, figsize=(15, 25))
    axes = axes.flatten()  # Flatten the grid for easier indexing

    # Plot normalized sensor readings in separate subplots
    for i, sensor in enumerate(sensors):
        axes[i].plot(df_engine['cycle'], df_engine[f'{sensor}_norm'], color='blue')

        # Add sensor name inside the graph (top-left corner)
        axes[i].text(0.05, 0.95, sensor, transform=axes[i].transAxes,
                     fontsize=5, verticalalignment='top', color='black', weight='bold')

        axes[i].set_xlabel('cucle', fontsize=8)
        axes[i].set_ylabel('Norm.Value', fontsize=7)

        # Reduce the size of x and y axis tick labels
        axes[i].tick_params(axis='both', which='major', labelsize=6)  # Set tick label size to 6
        axes[i].grid(True)

    # Remove any empty subplots
    for i in range(len(sensors), len(axes)):
        fig.delaxes(axes[i])

    # Adjust layout to prevent overlap
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    # Add a main title for the whole figure
    fig.suptitle(f'Normalized Sensor Readings for Engine {engine_id}', fontsize=6)

    # Show the plot
    plt.show()
