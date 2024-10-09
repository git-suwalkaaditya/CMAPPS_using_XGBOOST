import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import _tree
import matplotlib.pyplot as plt
from sklearn import tree

# Load your dataset
csv_path = 'F:\CMAPPS using XGBOOST\Book1.csv'
columns_to_use = ['F/NF', 'sensor_2', 'sensor_3', 'sensor_4', 'sensor_7', 'sensor_8', 'sensor_9', 'sensor_11',
                  'sensor_12', 'sensor_13', 'sensor_14', 'sensor_15', 'sensor_17', 'sensor_20', 'sensor_21']
num_rows = 480
df = pd.read_csv(csv_path, usecols=columns_to_use, nrows=num_rows)

# Prepare features (X) and target variable (y)
X = df.drop(columns=['F/NF'])
y = df['F/NF']


# Discretize the sensor readings using KBinsDiscretizer
discretizer = KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='uniform')
X_discretized = discretizer.fit_transform(X)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_discretized, y, test_size=0.2, random_state=42)
print("Training set class distribution:")
print(y_train.value_counts())
print("Test set class distribution:")
print(y_test.value_counts())

# Train a Gradient Boosting Decision Tree Classifier
gbdt = GradientBoostingClassifier(n_estimators=10, random_state=42)
gbdt.fit(X_train, y_train)


# Function to print node details and sample indices
def print_tree_node_details(tree, feature_names, X_train, y_train):
    tree_ = tree.tree_
    feature_names = [feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
                     for i in tree_.feature]

    # Get the indices of samples reaching each node
    node_indices = tree.apply(X_train)

    print("\nNode details for this tree:")
    for i in range(tree_.node_count):
        if tree_.children_left[i] == _tree.TREE_LEAF:
            print(f"Leaf Node {i}:")
            print(f"  Number of samples = {tree_.n_node_samples[i]}")
            print(f"  Values = {tree_.value[i]}")
            print(f"  Predicted class = {np.argmax(tree_.value[i])} (0: Healthy, 1: Faulty)")

            # Get the indices of samples reaching this leaf node
            leaf_samples_indices = np.where(node_indices == i)[0]
            print(f"  Sample indices reaching this leaf: {leaf_samples_indices}")

            # Print the actual samples reaching this leaf
            print(f"  Samples reaching this leaf:")
            for idx in leaf_samples_indices:
                print(f"    Sample {idx}: Features = {X_train[idx]}, Label = {y_train.iloc[idx]}")
        else:
            print(f"Node {i}:")
            print(f"  Decision rule: ({feature_names[i]} <= {tree_.threshold[i]})")
            print(f"  Number of samples = {tree_.n_node_samples[i]}")
            print(f"  Values = {tree_.value[i]}")
            print(f"  Predicted class = {np.argmax(tree_.value[i])} (0: Healthy, 1: Faulty)")


# Visualize each tree and print node details
for i, estimator in enumerate(gbdt.estimators_.flatten()):
    print(f"\nVisualizing Decision Tree {i + 1}:")

    # Print the tree node details (samples at each node and their classification)
    print_tree_node_details(estimator, [f"sensor_{i + 1}" for i in range(X.shape[1])], X_train, y_train)

    # Plot the decision tree
    plt.figure(figsize=(20, 10))
    tree.plot_tree(estimator,
                   feature_names=[f"sensor_{i + 1}" for i in range(X.shape[1])],
                   class_names=['Healthy', 'Faulty'],
                   filled=True,
                   rounded=True,
                   proportion=False)
    plt.title(f"Decision Tree {i + 1}")
    plt.show()