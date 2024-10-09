import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

# Path to your CSV file
csv_path = r'F:\CMAPPS using XGBOOST\Book1.csv'

# Specify the columns you want to read
columns_to_use = ['RUL', 'sensor_2', 'sensor_3', 'sensor_4', 'sensor_7', 'sensor_8', 'sensor_9', 'sensor_11', 'sensor_12', 'sensor_13', 'sensor_14', 'sensor_15', 'sensor_17', 'sensor_20', 'sensor_21']

# Specify the number of rows to read
num_rows = 450  # Adjust based on how many rows you want to read

# Load specific columns and rows from the CSV file
df = pd.read_csv(csv_path, usecols=columns_to_use, nrows=num_rows)

# Display the first few rows to confirm the data
print(df.head())

# Prepare features (X) and target variable (y)
X = df.drop(columns=['F/NF'])
y = df['F/NF']

# Split data into training and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the XGBoost classifier model
xgb_model = XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)

# Train the model
try:
    xgb_model.fit(X_train, y_train)
except Exception as e:
    print(f"Error during model training: {e}")

# Make predictions on the test data
y_pred = None  # Initialize y_pred to handle cases where prediction fails
try:
    y_pred = xgb_model.predict(X_test)
except Exception as e:
    print(f"Error during prediction: {e}")

# Only evaluate if y_pred is defined
if y_pred is not None:
    # Evaluate the model's performance
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)

    print(f"Accuracy: {accuracy}")
    print(f"Confusion Matrix:\n{conf_matrix}")

# Path to your new data CSV file
new_data_path = r'F:\CMAPPS using XGBOOST\Book2.csv'

# Load the new data
new_data = pd.read_csv(new_data_path, usecols=columns_to_use[1:])  # No 'F/NF'

# Use the trained model to make predictions
try:
    predictions = xgb_model.predict(new_data)
except Exception as e:
    print(f"Error during prediction on new data: {e}")

# Display predictions (0 and 1)
if 'predictions' in locals():
    print("Predictions for the new data (0 and 1):")
    print(predictions)
