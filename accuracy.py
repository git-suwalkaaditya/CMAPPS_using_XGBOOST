import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load the training dataset
csv_path_train = 'C:/Users\hp\Downloads\CMAPSSData/train_FD001.csv'
columns_to_use = ['RUL', 'sensor_2', 'sensor_3', 'sensor_4', 'sensor_7', 'sensor_8', 'sensor_9', 'sensor_11',
                  'sensor_12', 'sensor_13', 'sensor_14', 'sensor_15', 'sensor_17', 'sensor_20', 'sensor_21']
num_rows = 300
df_train = pd.read_csv(csv_path_train, usecols=columns_to_use, nrows=num_rows)

# Prepare features (X) and target variable (y) for training
X_train = df_train.drop(columns=['RUL'])
y_train = df_train['RUL']

# Normalize the features using StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Load the test dataset
csv_path_test = 'F:/CMAPPS using XGBOOST/test_engine_13.csv'
df_test = pd.read_csv(csv_path_test, usecols=columns_to_use[1:])  # Exclude 'RUL' for the test set

# Prepare the test set
X_test_scaled = scaler.transform(df_test)

# Load the test labels
csv_path_test_labels = 'F:/CMAPPS using XGBOOST/test_engine_13.csv'  # Assuming labels are in a separate file
df_test_labels = pd.read_csv(csv_path_test_labels, usecols=['RUL'])
y_test = df_test_labels['RUL']

# Initialize and train the XGBoost regressor
xgb_reg = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42)
xgb_reg.fit(X_train_scaled, y_train)

# Predict RUL for the test set
y_pred = xgb_reg.predict(X_test_scaled)

# Calculate Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("Mean Absolute Error (MAE):", mae)
print("Root Mean Squared Error (RMSE):", rmse)

results = pd.DataFrame({
    'Actual RUL': y_test,
    'Predicted RUL': y_pred
})

print("\nActual vs Predicted RUL:")
print(results.head(26))  # Display the first 10 results
