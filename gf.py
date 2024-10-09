import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset (update the path to your CSV file location)
df = pd.read_csv("C:/Users/hp/Downloads/CMAPSSData/train_FD001.csv")

# Specify the sensor to plot
sensor = 'sensor_2'

# List of engines to process (1 to 5)
engines = [1, 2, 3, 4, 5]

# Loop through the first 5 engines
for engine_id in engines:
    # Filter data for the specific engine
    df_engine = df[df['engine_id'] == engine_id]

    if df_engine.empty:
        print(f"No data found for engine {engine_id}. Skipping...")
        continue

    # Select the first 75 readings (rows)
    df_engine = df_engine.head(75)

    print(f"Processing Engine {engine_id} with {len(df_engine)} data points.")

    # Normalize the sensor data using z-score normalization for each engine separately
    if sensor in df_engine.columns:
        mu = df_engine[sensor].mean()
        sigma = df_engine[sensor].std()

        # Apply the normalization formula: 2 * (x - mean) / std - 1
        df_engine[f'{sensor}_norm'] = 2 * (df_engine[sensor] - mu) / sigma - 1

        # Create a plot for sensor 1
        plt.figure(figsize=(10, 5))
        plt.plot(df_engine['cycle'], df_engine[f'{sensor}_norm'], color='blue')

        # Add sensor name inside the graph (top-left corner)
        plt.text(0.05, 0.95, sensor, transform=plt.gca().transAxes,
                 fontsize=10, verticalalignment='top', color='black', weight='bold')

        # Set labels and title
        plt.xlabel('Cycle', fontsize=10)
        plt.ylabel('Normalized Value', fontsize=10)
        plt.title(f'Normalized {sensor} Readings (First 75) for Engine {engine_id}', fontsize=12)

        # Adjust layout and show the plot
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    else:
        print(f"Sensor column '{sensor}' not found in the dataset.")
