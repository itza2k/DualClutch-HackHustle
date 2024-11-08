import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
import os

# Set up the file path
dirname = os.path.dirname(__file__)
filename = os.path.join(dirname, 'boarders2.csv')

# Load data
data = pd.read_csv(filename)
scaler = MinMaxScaler(feature_range=(0, 1))

# Prepare data for LSTM
def create_sequences(data, seq_length=24):
    sequences = []
    labels = []
    for i in range(len(data) - seq_length):
        seq = data[i:i+seq_length]
        label = data[i + seq_length][0]  # The boarders at the next hour
        sequences.append(seq)
        labels.append(label)
    return np.array(sequences), np.array(labels)

# Scale and prepare sequences
features = data[["hour", "day_of_week", "is_weekend", "boarders"]]
scaled_features = scaler.fit_transform(features)
seq_length = 24  # Number of hours in one sequence

X, y = create_sequences(scaled_features, seq_length)

# Reshape input to (samples, timesteps, features) for LSTM
X = np.reshape(X, (X.shape[0], seq_length, X.shape[2]))

# Build LSTM model
model = Sequential([
    LSTM(50, activation="relu", input_shape=(seq_length, X.shape[2])),
    Dense(1)
])
model.compile(optimizer=Adam(learning_rate=0.001), loss="mean_squared_error")

# Train initial model
model.fit(X, y, epochs=10, batch_size=16, verbose=1)
print("Initial LSTM model trained.")


import time

# Function to update model with new data
def update_model(new_data, model, scaler, seq_length=24):
    # Scale new data
    new_scaled = scaler.transform(new_data)

    # Prepare sequences for updating the model
    X_new, y_new = create_sequences(new_scaled, seq_length)
    X_new = np.reshape(X_new, (X_new.shape[0], seq_length, X_new.shape[2]))

    # Update model with new data
    model.fit(X_new, y_new, epochs=1, batch_size=1, verbose=1)
    print("Model updated with new data.")

    # Evaluate performance
    predictions = model.predict(X_new)
    mae = np.mean(np.abs(predictions - y_new))
    print(f"Updated model MAE on new data: {mae:.2f}")

# Simulate live data entry and retraining
for i in range(5):  # Simulating 5 new data points coming in
    # Generate synthetic new data row
    new_hour = np.random.randint(0, 24)
    new_day_of_week = np.random.randint(0, 7)
    new_is_weekend = 1 if new_day_of_week >= 5 else 0
    new_boarders = 20 if new_is_weekend else 30
    new_boarders *= 2 if 7 <= new_hour <= 9 or 16 <= new_hour <= 18 else 1
    new_boarders += np.random.randint(-5, 6)

    # Create a new DataFrame row
    new_data = pd.DataFrame([[new_hour, new_day_of_week, new_is_weekend, new_boarders]],
                            columns=["hour", "day_of_week", "is_weekend", "boarders"])

    print(f"\nNew data: {new_data.iloc[0].to_dict()}")
    update_model(new_data, model, scaler)

    # Wait to simulate delay for live updates
    time.sleep(1)
