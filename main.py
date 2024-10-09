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

    # Plot the normalized sensor readings over time for this engine (for each sensor separately)
    for sensor in sensors:
        plt.figure(figsize=(10, 6))
        plt.plot(df_engine['cycle'], df_engine[f'{sensor}_norm'], label=f'{sensor}_norm')
        plt.xlabel('Time Cycle')
        plt.ylabel('Normalized Sensor Reading')
        plt.title(f'Normalized {sensor} Readings Over Time for Engine {engine_id}')
        plt.grid(True)
        plt.legend()
        plt.show()
