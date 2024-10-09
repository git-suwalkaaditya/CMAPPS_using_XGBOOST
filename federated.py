import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Client class for federated learning simulation
class Client:
    def __init__(self, data, target):
        self.X_train = data
        self.y_train = target
        self.local_trees = []

    def train_local_model(self, max_depth=3):
        # Train multiple decision trees locally
        self.local_trees = []
        for depth in range(1, max_depth + 1):
            tree = DecisionTreeClassifier(max_depth=depth)
            tree.fit(self.X_train, self.y_train)
            self.local_trees.append(tree)

    def get_local_tree_probabilities(self):
        # Calculate probabilities of each tree based on accuracy on local data
        tree_probabilities = []
        for tree in self.local_trees:
            y_pred = tree.predict(self.X_train)
            accuracy = accuracy_score(self.y_train, y_pred)
            tree_probabilities.append(accuracy)
        # Normalize probabilities
        tree_probabilities = np.array(tree_probabilities)
        tree_probabilities /= tree_probabilities.sum()
        return tree_probabilities, self.local_trees

# Server class to aggregate local models
class Server:
    def __init__(self):
        self.global_tree_probabilities = None
        self.global_trees = None

    def aggregate_models(self, client_models):
        all_trees = []
        all_probabilities = []
        for probs, trees in client_models:
            all_probabilities.append(probs)
            all_trees.append(trees)

        # Average the probabilities across clients
        avg_probabilities = np.mean(all_probabilities, axis=0)

        # Use the trees from the first client for simplicity
        self.global_tree_probabilities = avg_probabilities
        self.global_trees = all_trees[0]  # Assume all clients use same structure

    def calculate_final_probability(self, X_test):
        final_probs = np.zeros(X_test.shape[0])
        for i, tree in enumerate(self.global_trees):
            tree_probs = tree.predict_proba(X_test)[:, 1]  # Probability of 'Faulty' (class 1)
            weighted_probs = tree_probs * self.global_tree_probabilities[i]
            final_probs += weighted_probs
        return final_probs

# Simulating federated learning with multiple clients
def federated_training(clients, server, X_test, num_rounds=1):
    for round_num in range(num_rounds):
        client_models = []
        for client in clients:
            client.train_local_model()
            client_probs = client.get_local_tree_probabilities()
            client_models.append(client_probs)

        # Server aggregates local models
        server.aggregate_models(client_models)

    # Server uses global model to make predictions on test data
    final_probs = server.calculate_final_probability(X_test)
    return final_probs

# Sample data loading for two clients
csv_path_train1 = 'F:/CMAPPS using XGBOOST/Client1.csv'
csv_path_train2 = 'F:/CMAPPS using XGBOOST/Client2.csv'

columns_to_use = ['F/NF', 'sensor_2', 'sensor_3', 'sensor_4', 'sensor_7', 'sensor_8', 'sensor_9', 'sensor_11',
                  'sensor_12', 'sensor_13', 'sensor_14', 'sensor_15', 'sensor_17', 'sensor_20', 'sensor_21']

# Client 1 data
df_train1 = pd.read_csv(csv_path_train1, usecols=columns_to_use)
X_train1 = df_train1.drop(columns=['F/NF'])
y_train1 = df_train1['F/NF']

# Client 2 data
df_train2 = pd.read_csv(csv_path_train2, usecols=columns_to_use)
X_train2 = df_train2.drop(columns=['F/NF'])
y_train2 = df_train2['F/NF']

# Initialize clients
client1 = Client(X_train1, y_train1)
client2 = Client(X_train2, y_train2)

# Initialize server
server = Server()

# Load the test dataset
csv_path_test = 'F:/CMAPPS using XGBOOST/Book4.csv'
columns_to_use1 = ['sensor_2', 'sensor_3', 'sensor_4', 'sensor_7', 'sensor_8', 'sensor_9', 'sensor_11',
                   'sensor_12', 'sensor_13', 'sensor_14', 'sensor_15', 'sensor_17', 'sensor_20', 'sensor_21']
df_test = pd.read_csv(csv_path_test, usecols=columns_to_use1)
X_test = df_test

# Perform federated training and prediction
final_probs = federated_training([client1, client2], server, X_test, num_rounds=1)

# Optional: Classify based on threshold (e.g., 0.5 for binary classification)
predicted_classes = (final_probs > 0.5).astype(int)

print("Final Bayesian Probabilities (for Faulty):", final_probs)
print("Predicted Classes:", predicted_classes)
