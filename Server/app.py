'''
from flask import Flask, jsonify, request
from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os
import datetime
import random
dirname = os.path.dirname(__file__)
file_path= os.path.join(dirname, 'boarders4.csv')
test_file_path = os.path.join(dirname, 'boarders4.csv')
initial_data_path = file_path

class BoardingPredictor:
    def __init__(self, initial_data_path=None):
        """
        Initialize the predictor with initial dataset
        """
        # Create a default dataset if no path is provided
        if initial_data_path is None or not os.path.exists(initial_data_path):
            print("No initial dataset found. Creating a default dataset.")
            self.data = self.create_comprehensive_default_dataset()
        else:
            self.data = self.load_data(initial_data_path)
        
        # Ensure data is not None
        if self.data is None:
            self.data = self.create_comprehensive_default_dataset()
        
        self.processed_data = self.preprocess_data(self.data)
        self.model, self.scaler = self.train_initial_model()

    def create_comprehensive_default_dataset(self):
        """
        Create a comprehensive default dataset with realistic variations
        """
        print("Creating a comprehensive default dataset")
        
        # Generate data for multiple months
        months = range(1, 13)
        days = range(1, 32)
        hours = range(24)
        
        data = []
        
        for month in months:
            for day in days:
                for hour in hours:
                    # Simulate realistic passenger variations
                    base_passengers = 50  # Base passenger count
                    
                    # Weekend factor
                    weekend_multiplier = 1.5 if day % 7 in [0, 6] else 1.0
                    
                    # Time of day factor
                    time_multiplier = (
                        0.5 if hour < 6 else  # Early morning
                        1.2 if 6 <= hour < 9 else  # Morning rush
                        1.5 if 9 <= hour < 12 else  # Late morning
                        1.0 if 12 <= hour < 16 else  # Afternoon
                        1.3 if 16 <= hour < 19 else  # Evening rush
                        0.7  # Late night
                    )
                    
                    # Season factor
                    season_multiplier = (
                        1.1 if month in [6, 7, 8] else  # Summer
                        0.9 if month in [12, 1, 2] else  # Winter
                        1.0  # Other seasons
                    )
                    
                    # Random variation
                    random_variation = random.uniform(0.8, 1.2)
                    
                    # Holiday factor
                    holiday_multiplier = 0.7 if month in [1, 12] and day in [1, 25, 26] else 1.0
                    
                    # Event factor (occasional events)
                    event_multiplier = 1.5 if random.random() < 0.05 else 1.0
                    
                    # Rain factor
                    rain_multiplier = 0.8 if random.random() < 0.2 else 1.0
                    
                    # Calculate total passengers
                    total_passengers = (
                        base_passengers * 
                        weekend_multiplier * 
                        time_multiplier * 
                        season_multiplier * 
                        random_variation * 
                        holiday_multiplier * 
                        event_multiplier * 
                        rain_multiplier
                    )
                    
                    data.append({
                        'month': month,
                        'day': day,
                        'hour': hour,
                        'weekend': 1 if day % 7 in [0, 6] else 0,
                        'rain': 1 if rain_multiplier < 1.0 else 0,
                        'holiday': 1 if month in [1, 12] and day in [1, 25, 26] else 0,
                        'event': 1 if event_multiplier > 1.0 else 0,
                        'bus_stop_1': max(0, total_passengers)
                    })
        
        return pd.DataFrame(data)

    def load_data(self, file_path):
        """
        Load data from a CSV file with comprehensive error handling
        """
        try:
            if not os.path.exists(file_path):
                print(f"File not found: {file_path}")
                return None

            data = pd.read_csv(file_path)
            
            required_columns = ['month', 'day', 'hour', 'weekend', 'rain', 'holiday', 'event', 'bus_stop_1']
            missing_columns = [col for col in required_columns if col not in data.columns]
            
            if missing_columns:
                print(f"Missing columns: {missing_columns}")
                return None

            return data
        
        except Exception as e:
            print(f"Error loading data: {e}")
            return None

    def preprocess_data(self, data):
        """
        Preprocess the data by handling missing values and creating necessary features
        """
        if data is None:
            data = self.create_comprehensive_default_dataset()

        data = data.dropna()
        return data

    def train_initial_model(self):
        """
        Train the initial machine learning model
        """
        try:
            features = ['month', 'day', 'hour', 'weekend', 'rain', 'holiday', 'event']
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

    def predict_next_day(self):
        """
        Predict passenger count for the next day
        """
        try:
            if self.model is None or self.scaler is None:
                self.model, self.scaler = self.train_initial_model()
            
            # Get current date
            today = datetime.date.today()
            next_day = today.day + 1
            next_month = today.month
            
            # Handle month rollover
            if next_day > 31:
                next_day = 1
                next_month += 1
                if next_month > 12:
                    next_month = 1
            
            # Create next day features for all 24 hours
            next_day_features = pd.DataFrame({
                'month': [next_month] * 24,
                'day': [next_day] * 24,
                'hour': list(range(24)),
                'weekend': [1 if datetime.date(today.year, next_month, next_day).weekday() in [5, 6] else 0] * 24,
                'rain': [0] * 24,
                'holiday': [0] * 24,
                'event': [0] * 24
            })
            
            next_day_features_scaled = self.scaler.transform(next_day_features)
            next_day_predictions = self.model.predict(next_day_features_scaled)
            
            return next_day_predictions
        
        except Exception as e:
            print(f"Error predicting next day: {e}")
            return [0] * 24

    def update_model(self, new_boarding_data):
        """
        Update the model with new boarding information
        """
        try:
            self.processed_data = pd.concat([self.processed_data, new_boarding_data], ignore_index=True)
            self.processed_data = self.preprocess_data(self.processed_data)
            self.model, self.scaler = self.train_initial_model()
        except Exception as e:
            print(f"Error updating model: {e}")

    def get_historical_demand(self, num_days=7):
        """
        Retrieve historical demand data
        """
        try:
            historical_demand = self.processed_data.groupby(['day', 'hour'])['bus_stop_1'].mean().reset_index()
            historical_demand = historical_demand.sort_values('day', ascending=False).head(num_days * 24)
            historical_data = []
            for _, row in historical_demand.iterrows():
                historical_data.append({
                    "time": f"{int(row['hour']):02d}:00", 
                    "day": int(row['day']),
                    "presentDemand": round(row['bus_stop_1'], 2),
                    "predictedDemand": None
                })
            return historical_data
        except Exception as e:
            print(f"Error retrieving historical demand: {e}")
            return []

app = Flask(__name__)
CORS(app)

global_predictor = BoardingPredictor(file_path)

def initialize_demand_data():
    """
    Initialize demand data with historical and predicted data
    """
    global global_predictor
    historical_demand = global_predictor.get_historical_demand()
    next_day_predictions = global_predictor.predict_next_day()
    predicted_demand = []
    for hour, prediction in enumerate(next_day_predictions):
        predicted_demand.append({
            "time": f"{hour:02d}:00", 
            "day": historical_demand[0]['day'] + 1 if historical_demand else datetime.date.today().day,
            "presentDemand": None, 
            "predictedDemand": round(prediction, 2)
        })
    return historical_demand + predicted_demand

demand_data = initialize_demand_data()

current_conditions = {
    "rain": 0,
    "event": 0
}

@app.route('/api/demand', methods=['GET'])
def get_demand():
    """
    Get current demand data including historical and predicted data
    """
    return jsonify(demand_data)

@app.route('/api/update-conditions', methods=['POST'])
def update_conditions():
    """
    Update environmental conditions (rain, event)
    """
    global current_conditions, demand_data, global_predictor
    new_conditions = request.json
    current_conditions['rain'] = new_conditions.get('rain', 0)
    current_conditions['event'] = new_conditions.get('event', 0)
    
    new_boarding_df = pd.DataFrame({
        'day': [datetime.date.today().day],
        'hour': [datetime.datetime.now().hour],
        'weekend': [1 if datetime.date.today().weekday() in [5, 6] else 0],
        'rain': [current_conditions['rain']],
        'holiday': [0],
        'event': [current_conditions['event']],
        'bus_stop_1': [new_conditions.get('passengers', 0)]
    })
    
    global_predictor.update_model(new_boarding_df)
    demand_data = initialize_demand_data()
    
    return jsonify({
        'message': 'Conditions updated successfully',
        'currentConditions': current_conditions,
        'updatedDemand': demand_data
    }), 200

@app.route('/api/current-conditions', methods=['GET'])
def get_current_conditions():
    """
    Get current environmental conditions
    """
    return jsonify(current_conditions)

if __name__ == '__main__':
    app.run(debug=True)
    
'''
'''
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
    
'''




from flask import Flask, jsonify, request
from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os
import datetime
import random

dirname = os.path.dirname(__file__)
file_path= os.path.join(dirname, 'boarders4.csv')
test_file_path = os.path.join(dirname, 'boarders4.csv')
initial_data_path = file_path


class BoardingPredictor:
    def __init__(self, initial_data_path=None):
        """
        Initialize the predictor with initial dataset
        """
        # Create a default dataset if no path is provided
        if initial_data_path is None or not os.path.exists(initial_data_path):
            print("No initial dataset found. Creating a default dataset.")
            self.data = self.create_comprehensive_default_dataset()
        else:
            self.data = self.load_data(initial_data_path)
        
        # Ensure data is not None
        if self.data is None:
            self.data = self.create_comprehensive_default_dataset()
        
        self.processed_data = self.preprocess_data(self.data)
        self.models = {}
        self.scalers = {}
        self.train_initial_models()

    def create_comprehensive_default_dataset(self):
        """
        Create a comprehensive default dataset with realistic variations for 4 bus stops
        """
        print("Creating a comprehensive default dataset")
        
        # Generate data for multiple months
        months = range(1, 13)
        days = range(1, 32)
        hours = range(24)
        
        data = []
        
        for month in months:
            for day in days:
                for hour in hours:
                    # Simulate realistic passenger variations
                    base_passengers = [50, 45, 55, 40]  # Different base for each bus stop
                    
                    # Weekend factor
                    weekend_multiplier = 1.5 if day % 7 in [0, 6] else 1.0
                    
                    # Time of day factor
                    time_multiplier = (
                        0.5 if hour < 6 else  # Early morning
                        1.2 if 6 <= hour < 9 else  # Morning rush
                        1.5 if 9 <= hour < 12 else  # Late morning
                        1.0 if 12 <= hour < 16 else  # Afternoon
                        1.3 if 16 <= hour < 19 else  # Evening rush
                        0.7  # Late night
                    )
                    
                    # Season factor
                    season_multiplier = (
                        1.1 if month in [6, 7, 8] else  # Summer
                        0.9 if month in [12, 1, 2] else  # Winter
                        1.0  # Other seasons
                    )
                    
                    # Random variation
                    random_variation = [random.uniform(0.8, 1.2) for _ in range(4)]
                    
                    # Holiday factor
                    holiday_multiplier = 0.7 if month in [1, 12] and day in [1, 25, 26] else 1.0
                    
                    # Event factor (occasional events)
                    event_multiplier = 1.5 if random.random() < 0.05 else 1.0
                    
                    # Rain factor
                    rain_multiplier = 0.8 if random.random() < 0.2 else 1.0
                    
                    # Calculate total passengers for each bus stop
                    total_passengers = []
                    for i in range(4):
                        passengers = (
                            base_passengers[i] * 
                            weekend_multiplier * 
                            time_multiplier * 
                            season_multiplier * 
                            random_variation[i] * 
                            holiday_multiplier * 
                            event_multiplier * 
                            rain_multiplier
                        )
                        total_passengers.append(max(0, passengers))
                    
                    data.append({
                        'month': month,
                        'day': day,
                        'hour': hour,
                        'weekend': 1 if day % 7 in [0, 6] else 0,
                        'rain': 1 if rain_multiplier < 1.0 else 0,
                        'holiday': 1 if month in [1, 12] and day in [1, 25, 26] else 0,
                        'event': 1 if event_multiplier > 1.0 else 0,
                        'bus_stop_1': total_passengers[0],
                        'bus_stop_2': total_passengers[1],
                        'bus_stop_3': total_passengers[2],
                        'bus_stop_4': total_passengers[3]
                    })
        
        return pd.DataFrame(data)

    def load_data(self, file_path):
        """
        Load data from a CSV file with comprehensive error handling
        """
        try:
            if not os.path.exists(file_path):
                print(f"File not found: {file_path}")
                return None

            data = pd.read_csv(file_path)
            
            required_columns = ['month', 'day', 'hour', 'weekend', 'rain', 'holiday', 'event', 
                                'bus_stop_1', 'bus_stop_2', 'bus_stop_3', 'bus_stop_4']
            missing_columns = [col for col in required_columns if col not in data.columns]
            
            if missing_columns:
                print(f"Missing columns: {missing_columns}")
                return None

            return data
        
        except Exception as e:
            print(f"Error loading data: {e}")
            return None

    def preprocess_data(self, data):
        """
        Preprocess the data by handling missing values and creating necessary features
        """
        if data is None:
            data = self.create_comprehensive_default_dataset()

        data = data.dropna()
        return data

    def train_initial_models(self):
        """
        Train initial machine learning models for each bus stop
        """
        try:
            features = ['month', 'day', 'hour', 'weekend', 'rain', 'holiday', 'event']
            bus_stops = ['bus_stop_1', 'bus_stop_2', 'bus_stop_3', 'bus_stop_4']
            
            for bus_stop in bus_stops:
                X = self.processed_data[features]
                y = self.processed_data[bus_stop]
                
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                
                model = RandomForestRegressor(n_estimators=100, random_state=42)
                model.fit(X_scaled, y)
                
                self.models[bus_stop] = model
                self.scalers[bus_stop] = scaler
        
        except Exception as e:
            print(f"Error training initial models: {e}")

    def predict_next_day(self, bus_stop):
        """
        Predict passenger count for the next day for a specific bus stop
        """
        try:
            if bus_stop not in self.models or bus_stop not in self.scalers:
                raise ValueError(f"No model found for {bus_stop}")
            
            # Get current date
            today = datetime.date.today()
            next_day = today.day + 1
            next_month = today.month
            
            # Handle month rollover
            if next_day > 31:
                next_day = 1
                next_month += 1
                if next_month > 12:
                    next_month = 1
            
            # Create next day features for all 24 hours
            next_day_features = pd.DataFrame({
                'month': [next_month] * 24,
                'day': [next_day] * 24,
                'hour': list(range(24)), 'weekend': [1 if datetime.date(today.year, next_month, next_day).weekday() in [5, 6] else 0] * 24,
                'rain': [0] * 24,
                'holiday': [0] * 24,
                'event': [0] * 24
            })
            
            next_day_features_scaled = self.scalers[bus_stop].transform(next_day_features)
            next_day_predictions = self.models[bus_stop].predict(next_day_features_scaled)
            
            return next_day_predictions
        
        except Exception as e:
            print(f"Error predicting next day for {bus_stop}: {e}")
            return [0] * 24

    def update_model(self, new_boarding_data):
        """
        Update the model with new boarding information for all bus stops
        """
        try:
            self.processed_data = pd.concat([self.processed_data, new_boarding_data], ignore_index=True)
            self.processed_data = self.preprocess_data(self.processed_data)
            self.train_initial_models()
        except Exception as e:
            print(f"Error updating models: {e}")

    def get_historical_demand(self, bus_stop, num_days=7):
        """
        Retrieve historical demand data for a specific bus stop
        """
        try:
            historical_demand = self.processed_data.groupby(['day', 'hour'])[bus_stop].mean().reset_index()
            historical_demand = historical_demand.sort_values('day', ascending=False).head(num_days * 24)
            historical_data = []
            for _, row in historical_demand.iterrows():
                historical_data.append({
                    "time": f"{int(row['hour']):02d}:00", 
                    "day": int(row['day']),
                    "presentDemand": round(row[bus_stop], 2),
                    "predictedDemand": None
                })
            return historical_data
        except Exception as e:
            print(f"Error retrieving historical demand for {bus_stop}: {e}")
            return []

app = Flask(__name__)
CORS(app)

global_predictor = BoardingPredictor(file_path)

def initialize_demand_data():
    """
    Initialize demand data with historical and predicted data for all bus stops
    """
    global global_predictor
    demand_data = {}
    for bus_stop in ['bus_stop_1', 'bus_stop_2', 'bus_stop_3', 'bus_stop_4']:
        historical_demand = global_predictor.get_historical_demand(bus_stop)
        next_day_predictions = global_predictor.predict_next_day(bus_stop)
        predicted_demand = []
        for hour, prediction in enumerate(next_day_predictions):
            predicted_demand.append({
                "time": f"{hour:02d}:00", 
                "day": historical_demand[0]['day'] + 1 if historical_demand else datetime.date.today().day,
                "presentDemand": None, 
                "predictedDemand": round(prediction, 2)
            })
        demand_data[bus_stop] = historical_demand + predicted_demand
    return demand_data

demand_data = initialize_demand_data()

current_conditions = {
    "rain": 0,
    "event": 0
}

@app.route('/api/demand/<bus_stop>', methods=['GET'])
def get_demand(bus_stop):
    """
    Get current demand data for a specific bus stop
    """
    if bus_stop not in ['bus_stop_1', 'bus_stop_2', 'bus_stop_3', 'bus_stop_4']:
        return jsonify({"error": "Invalid bus stop"}), 400
    return jsonify(demand_data[bus_stop])

@app.route('/api/update-conditions', methods=['POST'])
def update_conditions():
    """
    Update environmental conditions (rain, event)
    """
    global current_conditions, demand_data, global_predictor
    new_conditions = request.json
    current_conditions['rain'] = new_conditions.get('rain', 0)
    current_conditions['event'] = new_conditions.get('event', 0)
    
    new_boarding_df = pd.DataFrame({
        'day': [datetime.date.today().day],
        'hour': [datetime.datetime.now().hour],
        'weekend': [1 if datetime.date.today().weekday() in [5, 6] else 0],
        'rain': [current_conditions['rain']],
        'holiday': [0],
        'event': [current_conditions['event']],
        'bus_stop_1': [new_conditions.get('passengers_bus_stop_1', 0)],
        'bus_stop_2': [new_conditions.get('passengers_bus_stop_2', 0)],
        'bus_stop_3': [new_conditions.get('passengers_bus_stop_3', 0)],
        'bus_stop_4': [new_conditions .get('passengers_bus_stop_4', 0)]
    })
    
    global_predictor.update_model(new_boarding_df)
    demand_data = initialize_demand_data()
    
    return jsonify({
        'message': 'Conditions updated successfully',
        'currentConditions': current_conditions,
        'updatedDemand': demand_data
    }), 200

@app.route('/api/current-conditions', methods=['GET'])
def get_current_conditions():
    """
    Get current environmental conditions
    """
    return jsonify(current_conditions)

if __name__ == '__main__':
    app.run(debug=True)