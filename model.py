import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle

# Load dataset (replace 'networkintrusion.csv' with your actual file name)
data = pd.read_csv('networkintrusion.csv')

# Selecting features and target
features = ['duration', 'protocol_type', 'src_bytes', 'dst_bytes']
X = data[features].copy()  # Ensure we create a proper copy to avoid warnings
y = data['class'].apply(lambda x: 1 if x == 'anomaly' else 0)  # 1 for 'anomaly', 0 for 'normal'

# Define the mapping for protocol_type
protocol_mapping = {'tcp': 0, 'udp': 1, 'icmp': 2}

# Map protocol_type and handle unmapped or missing values
X['protocol_type'] = X['protocol_type'].map(protocol_mapping)
if X['protocol_type'].isna().any():
    print("Warning: Unmapped or missing values found in 'protocol_type'. Replacing with default value 0 (tcp).")
    X['protocol_type'].fillna(0, inplace=True)  # Replace NaN or unmapped values with a default

# Handle missing values in other columns if any
X.fillna(0, inplace=True)

# Ensure all columns are numeric
X = X.apply(pd.to_numeric, errors='coerce').fillna(0)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize the RandomForest model
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Make predictions and calculate accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Save the trained model
with open("model.pkl", "wb") as model_file:
    pickle.dump(model, model_file)

