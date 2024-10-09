import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Load the training dataset
csv_path_train = 'F:/CMAPPS using XGBOOST/Book1.csv'
columns_to_use = ['F/NF', 'sensor_2', 'sensor_3', 'sensor_4', 'sensor_7', 'sensor_8', 'sensor_9', 'sensor_11',
                  'sensor_12', 'sensor_13', 'sensor_14', 'sensor_15', 'sensor_17', 'sensor_20', 'sensor_21']
num_rows = 470
df_train = pd.read_csv(csv_path_train, usecols=columns_to_use, nrows=num_rows)

# Prepare features (X) and target variable (y) for training
X_train = df_train.drop(columns=['F/NF'])
y_train = df_train['F/NF']

# Load the test dataset
csv_path_test = 'F:/CMAPPS using XGBOOST/Book4.csv'
columns_to_use1 = ['sensor_2', 'sensor_3', 'sensor_4', 'sensor_7', 'sensor_8', 'sensor_9', 'sensor_11',
                   'sensor_12', 'sensor_13', 'sensor_14', 'sensor_15', 'sensor_17', 'sensor_20', 'sensor_21']
df_test = pd.read_csv(csv_path_test, usecols=columns_to_use1)

# Prepare the test set
X_test = df_test

# Function to generate all possible decision trees (simplified for demonstration)
def generate_all_possible_trees(X_train, y_train, max_depth=3):
    trees = []
    for depth in range(1, max_depth + 1):
        tree = DecisionTreeClassifier(max_depth=depth)
        tree.fit(X_train, y_train)
        trees.append(tree)
    return trees

# Generate all possible decision trees
trees = generate_all_possible_trees(X_train, y_train)

# Function to calculate the probability of each tree being the ideal candidate
def calculate_tree_probabilities(trees, X_train, y_train):
    tree_probabilities = []
    for tree in trees:
        y_pred = tree.predict(X_train)
        accuracy = accuracy_score(y_train, y_pred)
        tree_probabilities.append(accuracy)
    # Normalize probabilities
    tree_probabilities = np.array(tree_probabilities)
    tree_probabilities /= tree_probabilities.sum()
    return tree_probabilities

# Calculate tree probabilities
tree_probabilities = calculate_tree_probabilities(trees, X_train, y_train)

# Function to calculate the final probability of x belonging to y1
def calculate_final_probability(trees, tree_probabilities, X_test):
    final_probs = np.zeros(X_test.shape[0])
    for i, tree in enumerate(trees):
        tree_probs = tree.predict_proba(X_test)[:, 1]  # Probability of 'Faulty' (class 1)
        weighted_probs = tree_probs * tree_probabilities[i]
        final_probs += weighted_probs
    return final_probs

# Calculate the final probability for the test set
final_probs = calculate_final_probability(trees, tree_probabilities, X_test)

print("Final Bayesian Probabilities (for Faulty):", final_probs)

# Optional: Classify based on threshold (e.g., 0.5 for binary classification)
predicted_classes = (final_probs > 0.5).astype(int)
print("Predicted Classes:", predicted_classes)