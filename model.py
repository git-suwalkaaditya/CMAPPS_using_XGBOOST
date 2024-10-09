import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

# Path to your CSV file
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

# Split data into training and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the XGBoost classifier model
xgb_model = XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=7, random_state=42)

# Train the model
xgb_model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = xgb_model.predict(X_test)

# Evaluate the model performance
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print(f"Confusion Matrix:\n{conf_matrix}")
