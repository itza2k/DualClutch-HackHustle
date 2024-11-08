import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os
import datetime
# Set up the file path
dirname = os.path.dirname(__file__)
file_path= os.path.join(dirname, 'boarders4.csv')
test_file_path = os.path.join(dirname, 'boarders4.csv')


class BoardingPredictor:
    def __init__(self, initial_data_path=None):
        """
        Initialize the predictor with initial dataset
        """
        # Create a default dataset if no path is provided
        if initial_data_path is None or not os.path.exists(initial_data_path):
            print("No initial dataset found. Creating a default dataset.")
            self.data = self.create_default_dataset()
        else:
            self.data = self.load_data(initial_data_path)
        
        # Ensure data is not None
        if self.data is None:
            self.data = self.create_default_dataset()
        
        self.processed_data = self.preprocess_data(self.data)
        self.model, self.scaler = self.train_initial_model()

    def create_default_dataset(self):
        """
        Create a default dataset if no initial data is available
        """
        print("Creating a default dataset for initial training")
        default_data = pd.DataFrame({
            'day': list(range(1, 31)) * 24,  # 30 days, repeated for 24 hours
            'hour': list(range(24)) * 30,
            'weekend': [1 if i % 7 in [5, 6] else 0 for i in range(30 * 24)],
            'rain': np.random.choice([0, 1], size=30*24, p=[0.8, 0.2]),
            'holiday': np.random.choice([0, 1], size=30*24, p=[0.95, 0.05]),
            'event': np.random.choice([0, 1], size=30*24, p=[0.9, 0.1]),
            'bus_stop_1': np.random.normal(50, 10, size=30*24)  # Random passenger counts
        })
        return default_data

    def load_data(self, file_path):
        """
        Load data from a CSV file with comprehensive error handling
        """
        try:
            # Check if file exists
            if not os.path.exists(file_path):
                print(f"File not found: {file_path}")
                return None

            # Attempt to read the file
            data = pd.read_csv(file_path)
            
            # Validate basic structure
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
        """
        Preprocess the data by handling missing values and creating necessary features
        """
        # Validate input
        if data is None:
            print("No data to preprocess. Using default dataset.")
            data = self.create_default_dataset()

        # Check for missing values
        print("Missing values:")
        print(data.isnull().sum())
        
        # Fill or drop missing values
        data = data.dropna()
        
        # Ensure all required columns exist
        required_columns = ['day', 'hour', 'weekend', 'rain', 'holiday', 'event', 'bus_stop_1']
        
        # Create columns if they don't exist
        for col in required_columns:
            if col not in data.columns:
                if col in ['weekend', 'holiday', 'rain', 'event']:
                    # Create binary columns
                    data[col] = 0
                elif col in ['day', 'hour']:
                    # Create time-based columns
                    data[col] = 0
        
        return data

    def train_initial_model(self):
        """
        Train the initial machine learning model
        """
        try:
            # Select features and target
            features = ['day', 'hour', 'weekend', 'rain', 'holiday', 'event']
            target = 'bus_stop_1'
            
            # Ensure all columns exist
            X = self.processed_data[features]
            y = self.processed_data[target]
            
            # Scale features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Train Random Forest Regressor
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_scaled, y)
            
            return model, scaler
        
        except Exception as e:
            print(f"Error training initial model: {e}")
            return None, None

    def create_next_day_features(self):
        """
        Create features for the next day prediction
        """
        # Get the last date in the dataset
        last_date = self.processed_data['day'].max()
        
        # Determine if it's a weekend
        try:
            # Use current date to determine weekend
            current_date = datetime.date.today()
            next_day = current_date.replace(day=current_date.day + 1)
            is_weekend = next_day.weekday() in [5, 6]
        except:
            # Fallback to simple weekend check
            is_weekend = 0
        
        # Create next day features for all 24 hours
        next_day_features = pd.DataFrame({
            'day': [last_date + 1] * 24,
            'hour': list(range(24)),
            'weekend': [1 if is_weekend else 0] * 24,
            'rain': [0] * 24,  # Default to no rain, can be adjusted
            'holiday': [0] * 24,  # Default to no holiday, can be adjusted
            'event': [0] * 24  # Default to no event, can be adjusted
        })
        
        return next_day_features

    def predict_next_day(self):
        """
        Predict passenger count for the next day
        """
        try:
            # Check if model and scaler are trained
            if self.model is None or self.scaler is None:
                print("Model not trained. Training initial model.")
                self.model, self.scaler = self.train_initial_model()
            
            # Create next day features
            next_day_features = self.create_next_day_features()
            
            # Scale next day features
            next_day_features_scaled = self.scaler.transform(next_day_features)
            
            # Predict for next day
            next_day_predictions = self.model.predict(next_day_features_scaled)
            
            # Print predictions
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
        """
        Update the model with new boarding information
        
        :param new_boarding_data: DataFrame with columns matching existing dataset
        """
        try:
            # Combine existing data with new boarding data
            self.processed_data = pd.concat([self.processed_data, new_boarding_data], ignore_index=True)
            
            # Select features and target
            features = ['day', 'hour', 'weekend', 'rain', 'holiday', 'event']
            target = 'bus_stop_1'
            
            # Ensure all columns exist
            X = self.processed_data[features]
            y = self.processed_data[target]
            
            # Scale features
            X_scaled = self.scaler.transform(X)
            
            # Retrain the model with updated data
            self.model.fit(X_scaled, y)
            print("Model updated with new boarding data.")
        
        except Exception as e:
            print(f"Error updating model: {e}")

def main():
    # Initialize the predictor 
    predictor = BoardingPredictor()  # No need to specify a file path
    
    # Initial prediction
    predictor.predict_next_day()
    
    # Simulate updating with new boarding data
    while True:
        # User interaction loop
        print("\nOptions:")
        print("1. Enter new boarding data")
        print("2. Predict next day")
        print("3. Exit")
        
        choice = input("Enter your choice (1-3): ")
        
        if choice == '1':
            # Collect new boarding data from user
            try:
                day = int(input("Enter day: "))
                hour = int(input("Enter hour (0-23): "))
                passengers = float(input("Enter number of passengers: "))
                
                # Create DataFrame with new boarding data
                new_data = pd.DataFrame({
                    'day': [day],
                    'hour': [hour],
                    'weekend': [1 if datetime.date(datetime.date.today().year, datetime.date.today().month, day).weekday() in [5, 6] else 0],
                    'rain': [0],  # You can modify this based on actual conditions
                    'holiday': [0],  # You can modify this based on actual conditions
                    'event': [0],  # You can modify this based on actual conditions
                    'bus_stop_1': [passengers]
                })
                
                # Update model with new data
                predictor.update_model(new_data)
            
            except ValueError:
                print("Invalid input. Please enter correct values.")
        
        elif choice == '2':
            # Predict for next day
            predictor.predict_next_day()
        
        elif choice == '3':
            print("Exiting the program. Goodbye!")
            break
        
        else:
            print("Invalid choice. Please select a valid option.")

if __name__ == "__main__":
    main()