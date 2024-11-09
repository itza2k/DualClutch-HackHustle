from flask import Flask, request, jsonify
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import datetime

app = Flask(__name__)

class BoardingPredictor:
    def __init__(self, initial_data_path=None):
        if initial_data_path is None or not os.path.exists(initial_data_path):
            print("No initial dataset found. Creating a default dataset.")
            self.data = self.create_default_dataset()
        else:
            self.data = self.load_data(initial_data_path)

        if self.data is None:
            self.data = self.create_default_dataset()

        self.processed_data = self.preprocess_data(self.data)
        self.model, self.scaler = self.train_initial_model()

    def create_default_dataset(self):
        print("Creating a default dataset for initial training")
        default_data = pd.DataFrame({
            'day': list(range(1, 31)) * 24,
            'hour': list(range(24)) * 30,
            'weekend': [1 if i % 7 in [5, 6] else 0 for i in range(30 * 24)],
            'rain': np.random.choice([0, 1], size=30*24, p=[0.8, 0.2]),
            'holiday': np.random.choice([0, 1], size=30*24, p=[0.95, 0.05]),
            'event': np.random.choice([0, 1], size=30*24, p=[0.9, 0.1]),
            'bus_stop_1': np.random.normal(50, 10, size=30*24)
        })
        return default_data

    def load_data(self, file_path):
        try:
            if not os.path.exists(file_path):
                print(f"File not found: {file_path}")
                return None

            data = pd.read_csv(file_path)
            required_columns = ['day', 'hour', 'weekend', 'rain', 'holiday', 'event', 'bus_stop_1']
            missing_columns = [col for col in required_columns if col not in data.columns]

            if missing_columns:
                print(f"Missing columns: {missing_columns}")
                print("Available columns:", list(data.columns))
                return None

            print("Columns in the dataset:")
            print(data.columns)
            return data

        except Exception as e:
            print(f"Error loading data: {e}")
            return None

    def preprocess_data(self, data):
        if data is None:
            print("No data to preprocess. Using default dataset.")
            data = self.create_default_dataset()

        print("Missing values:")
        print(data.isnull().sum())
        data = data.dropna()
        required_columns = ['day', 'hour', 'weekend', 'rain', 'holiday', 'event', 'bus_stop_1']

        for col in required_columns:
            if col not in data.columns:
                if col in ['weekend', 'holiday', 'rain', 'event']:
                    data[col] = 0
                elif col in ['day', 'hour']:
                    data[col] = 0

        return data

    def train_initial_model(self):
        try:
            features = ['day', 'hour', 'weekend', 'rain', 'holiday', 'event']
            target = 'bus_stop_1'
            X = self.processed_data[features]
            y = self.processed_data[target]
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_scaled, y)
            return model, scaler

        except Exception as e:
            print(f"Error training initial model: {e}")
            return None, None

    def create_next_day_features(self):
        last_date = self.processed_data['day'].max()
        try:
            current_date = datetime.date.today()
            next_day = current_date.replace(day=current_date.day + 1)
            is_weekend = next_day.weekday() in [5, 6]
        except:
            is_weekend = 0

        next_day_features = pd.DataFrame({
            'day': [last_date + 1] * 24,
            'hour': list(range(24)),
            'weekend': [1 if is_weekend else 0] * 24,
            'rain': [0] * 24,
            'holiday': [0] * 24,
            'event': [0] * 24
        })

        return next_day_features

    def predict_next_day(self):
        try:
            if self.model is None or self.scaler is None:
                print("Model not trained. Training initial model.")
                self.model, self.scaler = self.train_initial_model()

            next_day_features = self.create_next_day_features()
            next_day_features_scaled = self.scaler.transform(next_day_features)
            next_day_predictions = self.model.predict(next_day_features_scaled)

            print("\n--- Next Day Predictions ---")
            print("Hour | Predicted Bus Stop 1 Passengers")
            print("-" * 40)
            for hour, prediction in zip(next_day_features['hour'], next_day_predictions):
                print(f"{hour:4d} | {prediction:6.2f}")

            return next_day_predictions

        except Exception as e:
            print(f"Error predicting next day: {e}")
            return None

    def update_model(self, new_boarding_data):
        try:
            self.processed_data = pd.concat([self.processed_data, new_boarding_data], ignore_index=True)
            features = ['day', 'hour', 'weekend', 'rain', 'holiday', 'event']
            target = 'bus_stop_1'
            X = self.processed_data[features]
            y = self.processed_data[target]
            X_scaled = self.scaler.transform(X)
            self.model.fit(X_scaled, y)
            print("Model updated with new boarding data.")

        except Exception as e:
            print(f"Error updating model: {e}")

# Set the path to your CSV file
csv_file_path = 'content/boarders4.csv'

# Initialize the predictor with the CSV file
predictor = BoardingPredictor(initial_data_path=csv_file_path)

@app.route('/predict', methods=['GET'])
def predict():
    predictions = predictor.predict_next_day()
    return jsonify(predictions.tolist())

@app.route('/update', methods=['POST'])
def update():
    data = request.get_json()
    new_data = pd.DataFrame(data)
    predictor.update_model(new_data)
    return jsonify({"status": "success"})

if __name__ == '__main__':
    app.run(debug=True)