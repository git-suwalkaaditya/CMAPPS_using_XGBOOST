import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import _tree
import matplotlib.pyplot as plt
from sklearn import tree

# Load the training dataset
csv_path_train = 'F:/CMAPPS using XGBOOST/Book1.csv'
columns_to_use = ['F/NF', 'sensor_2', 'sensor_3', 'sensor_4', 'sensor_7', 'sensor_8', 'sensor_9', 'sensor_11',
                  'sensor_12', 'sensor_13', 'sensor_14', 'sensor_15', 'sensor_17', 'sensor_20', 'sensor_21']
num_rows = 480
df_train = pd.read_csv(csv_path_train, usecols=columns_to_use, nrows=num_rows)

# Prepare features (X) and target variable (y) for training
X_train = df_train.drop(columns=['F/NF'])
y_train = df_train['F/NF']

# Discretize the sensor readings using KBinsDiscretizer
discretizer = KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='uniform')
X_train_discretized = discretizer.fit_transform(X_train)

# Train the Gradient Boosting Decision Tree (GBDT) model
gbdt = GradientBoostingClassifier(n_estimators=10, random_state=42)
gbdt.fit(X_train_discretized, y_train)

# Load the test dataset
csv_path_test = 'F:/CMAPPS using XGBOOST/Book2.csv'
# Replace with your test data file path
columns_to_use1 = [ 'sensor_2', 'sensor_3', 'sensor_4', 'sensor_7', 'sensor_8', 'sensor_9', 'sensor_11',
                  'sensor_12', 'sensor_13', 'sensor_14', 'sensor_15', 'sensor_17', 'sensor_20','sensor_21']
df_test = pd.read_csv(csv_path_test, usecols=columns_to_use1)  # Exclude the target column

# Prepare features (X) for testing
X_test = df_test

# Discretize the test data using the same discretizer
X_test_discretized = discretizer.transform(X_test)

# Calculate the prior probability of being faulty
total_samples = len(y_train)
total_faulty_samples = sum(y_train)
prior_prob_faulty = total_faulty_samples / total_samples
prior_prob_healthy = 1 - prior_prob_faulty

# Function to print node details and handle different node shapes
def print_tree_node_details(tree, feature_names):
    tree_ = tree.tree_
    feature_names = [feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
                     for i in tree_.feature]

    print("\nNode details for this tree:")
    for i in range(tree_.node_count):
        node_samples = tree_.value[i][0]  # This gets the sample count for each class at the node

        # Handle cases where only one class is present in the node
        if len(node_samples) == 1:
            healthy_samples = node_samples[0] if y_train.unique()[0] == 0 else 0  # Class 0 (Healthy)
            faulty_samples = node_samples[0] if y_train.unique()[0] == 1 else 0  # Class 1 (Faulty)
        else:
            healthy_samples = node_samples[0]  # Class 0 (Healthy)
            faulty_samples = node_samples[1]  # Class 1 (Faulty)

        total_samples_at_node = healthy_samples + faulty_samples
        prob_faulty = faulty_samples / total_samples_at_node if total_samples_at_node > 0 else 0
        prob_healthy = healthy_samples / total_samples_at_node if total_samples_at_node > 0 else 0

        # Bayesian update
        bayesian_prob_faulty = (prob_faulty * prior_prob_faulty) / ((prob_faulty * prior_prob_faulty) + (prob_healthy * prior_prob_healthy))
        bayesian_prob_healthy = 1 - bayesian_prob_faulty

        if tree_.children_left[i] == _tree.TREE_LEAF:
            print(f"Leaf Node {i}:")
            print(f"  Number of samples = {tree_.n_node_samples[i]}")
            print(f"  Values (Healthy, Faulty) = [{healthy_samples}, {faulty_samples}]")  # Display counts for both classes
            print(f"  Conditional Probability of Faulty = {prob_faulty:.2f}")
            print(f"  Conditional Probability of Healthy = {prob_healthy:.2f}")
            print(f"  Bayesian Probability of Faulty = {bayesian_prob_faulty:.2f}")
            print(f"  Bayesian Probability of Healthy = {bayesian_prob_healthy:.2f}")
        else:
            print(f"Node {i}:")
            print(f"  Decision rule: ({feature_names[i]} <= {tree_.threshold[i]})")
            print(f"  Number of samples = {tree_.n_node_samples[i]}")
            print(f"  Values (Healthy, Faulty) = [{healthy_samples}, {faulty_samples}]")  # Display counts for both classes
            print(f"  Conditional Probability of Faulty = {prob_faulty:.2f}")
            print(f"  Conditional Probability of Healthy = {prob_healthy:.2f}")
            print(f"  Bayesian Probability of Faulty = {bayesian_prob_faulty:.2f}")
            print(f"  Bayesian Probability of Healthy = {bayesian_prob_healthy:.2f}")

# Visualize each tree and print node details
all_probs = []
for i, estimator in enumerate(gbdt.estimators_.flatten()):
    print(f"\nVisualizing Decision Tree {i + 1}:")

    # Print the tree node details (samples at each node and their classification)
    print_tree_node_details(estimator, [f"sensor_{i + 1}" for i in range(X_train.shape[1])])

    # Predict probability of being faulty (class 1)
    tree_probs = estimator.predict_proba(X_test_discretized)[:, 1]
    all_probs.append(tree_probs)

    # Plot the decision tree
    plt.figure(figsize=(20, 10))
    tree.plot_tree(estimator,
                   feature_names=[f"sensor_{i + 1}" for i in range(X_train.shape[1])],
                   class_names=['Healthy', 'Faulty'],
                   filled=True,
                   rounded=True,
                   proportion=False)  # Show actual sample counts
    plt.title(f"Decision Tree {i + 1}")
    plt.show()

# Convert to numpy array
all_probs = np.array(all_probs)

# Calculate the average of probabilities
final_probs = np.mean(all_probs, axis=0)

print("Final Probabilities:", final_probs)