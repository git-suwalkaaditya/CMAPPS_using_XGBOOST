import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load your dataset (update with the correct path to your dataset)
df = pd.read_csv("C:/Users/hp/Downloads/CMAPSSData/train_FD001.csv")

# Assuming 'engine_id' and 'cycle' columns are present in the dataset
# Get the maximum cycle for each engine (i.e., the engine's lifecycle)
engine_lifecycles = df.groupby('engine_id')['cycle'].max()

# Sort engines by their ID (just to ensure proper order)
engine_lifecycles = engine_lifecycles.sort_index()

# Define the segments and their corresponding colors
segments = [15, 30, 20, 20, 15]
colors = ['skyblue', 'lightgreen', 'orange', 'yellow', 'pink']

# Create a bar plot showing the lifecycles of each engine
plt.figure(figsize=(15, 7))

# Initialize the starting index for the segments
start_index = 0

# Plot each segment with its corresponding color
for segment, color in zip(segments, colors):
    end_index = start_index + segment
    plt.barh(engine_lifecycles.index[start_index:end_index], engine_lifecycles.values[start_index:end_index], color=color, edgecolor='black')
    start_index = end_index

# Add vertical lines for each "year" (i.e., every 50 cycles)
for cycle in range(50, max(engine_lifecycles.values) + 50, 50):
    plt.axvline(x=cycle, color='black', linestyle='--', linewidth=0.8, alpha=0.7)
    plt.text(cycle, -3, f'{cycle // 50} year', color='black', horizontalalignment='center')

# Set x and y labels
plt.xlabel('Number of Cycles (Lifecycle)', fontsize=12)
plt.ylabel('Engine ID', fontsize=12)

# Set plot title
plt.title('Engine Lifecycles with Yearly Markers', fontsize=16)

# Show the grid and plot
plt.grid(True, axis='x', linestyle='--', linewidth=0.5)
plt.tight_layout()

# Show the plot
plt.show()