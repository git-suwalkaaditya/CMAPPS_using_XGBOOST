import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import tree
import matplotlib.pyplot as plt
import graphviz

# Load the dataset
csv_path = 'F:\CMAPPS using XGBOOST\Book1.csv'

# Specify the columns you want to read (e.g., 'cycle', 'sensor_1', 'sensor_2', etc.)
columns_to_use = ['F/NF',  'sensor_2', 'sensor_3', 'sensor_4', 'sensor_7','sensor_8', 'sensor_9', 'sensor_11', 'sensor_12' , 'sensor_13', 'sensor_14', 'sensor_15', 'sensor_17', 'sensor_20', 'sensor_21' ]  # 'target' is your label

# Specify the number of rows to read
num_rows = 480  # Adjust this based on how many rows you want to read

# Load specific columns and rows from the CSV file
df = pd.read_csv(csv_path, usecols=columns_to_use, nrows=num_rows)

# Display the first few rows to confirm the data
print(df.head())

# Prepare features (X) and target variable (y)
X = df.drop(columns=['F/NF'])  # Dropping the target column
y = df['F/NF']  # Target column (this should be your classification label)

# Assuming your data is loaded in 'df', with 'label' as the target column and sensor readings in 'sensor_1', 'sensor_2', ...



# Step 1: Discretize the sensor readings using KBinsDiscretizer
# You can adjust the number of bins and strategy as per the distribution of your data
discretizer = KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='uniform')
X_discretized = discretizer.fit_transform(X)

# Step 2: Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_discretized, y, test_size=0.2, random_state=42)

# Step 3: Train a Gradient Boosting Decision Tree Classifier
gbdt = GradientBoostingClassifier(n_estimators=10, random_state=42)
gbdt.fit(X_train, y_train)

# Step 4: Visualize the decision trees
for i, estimator in enumerate(gbdt.estimators_.flatten()):
    # Export decision tree to Graphviz format
    dot_data = tree.export_graphviz(estimator, out_file=None,
                                    feature_names=X.columns,
                                    class_names=['Healthy', 'Faulty'],
                                    filled=True, rounded=True,
                                    special_characters=True)

    # Visualize with Graphviz
    graph = graphviz.Source(dot_data)
    graph.render(f"tree_{i}")  # This will save the trees as PDF files

    # Alternatively, you can visualize the trees inline in Jupyter Notebook by using
    # graph.view()

# Optional: Visualize decision tree split values and how many points reach each node
# You can use matplotlib for inline visualization
plt.figure(figsize=(20, 10))
tree.plot_tree(gbdt.estimators_[0][0], feature_names=X.columns, class_names=['Healthy', 'Faulty'], filled=True)
plt.show()
