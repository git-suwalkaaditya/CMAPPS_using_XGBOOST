import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Path to your CSV file
csv_path = r'F:\CMAPPS using XGBOOST\Book3.csv'

# Specify the columns to use, including the RUL column
columns_to_use = ['RUL', 'sensor_2', 'sensor_3', 'sensor_4', 'sensor_7', 'sensor_8', 'sensor_9', 'sensor_11',
                  'sensor_12', 'sensor_13', 'sensor_14', 'sensor_15', 'sensor_17', 'sensor_20', 'sensor_21']

 #Load specific columns from the CSV file
df = pd.read_csv(csv_path, usecols=columns_to_use)

 # Display the first few rows to confirm the data
 print(df.head())

# Prepare features (X) and target variable (y)
X = df.drop(columns=['RUL'])  # Drop the RUL column
y = df['RUL']  # RUL column

# Split data into training and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the XGBoost regressor model
xgb_model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)

# Train the model
try:
    xgb_model.fit(X_train, y_train)
except Exception as e:
    print(f"Error during model training: {e}")

# Make predictions on the test data
try:
    y_pred = xgb_model.predict(X_test)
except Exception as e:
    print(f"Error during prediction: {e}")

# Evaluate the model's performance
if 'y_pred' in locals():
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)

    print(f"Mean Absolute Error: {mae}")
    print(f"Mean Squared Error: {mse}")

# Path to your new data CSV file
new_data_path = r'F:\CMAPPS using XGBOOST\Book2.csv'

# Load the new data for predictions
new_data = pd.read_csv(new_data_path, usecols=columns_to_use[1:])  # Exclude 'RUL'

# Use the trained model to make predictions
try:
    predictions = xgb_model.predict(new_data)
except Exception as e:
    print(f"Error during prediction on new data: {e}")

# Display predictions for RUL
if 'predictions' in locals():
    print("Predictions for the new data (RUL):")
    print(predictions)
