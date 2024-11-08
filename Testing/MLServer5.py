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

def load_data(file_path):
    """
    Load data from a CSV file with error handling
    """
    try:
        data = pd.read_csv(file_path)
        
        # Verify columns
        print("Columns in the dataset:")
        print(data.columns)
        
        return data
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def preprocess_data(data):
    """
    Preprocess the data by handling missing values and creating necessary features
    """
    # Check for missing values
    print("Missing values:")
    print(data.isnull().sum())
    
    # Fill missing values or drop rows
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

def create_next_day_features(data):
    """
    Create features for the next day prediction
    """
    # Get the last date in the dataset
    last_date = data['day'].max()
    
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

def train_and_predict(train_data):
    """
    Train a machine learning model and make predictions
    """
    try:
        # Select features and target
        features = ['day', 'hour', 'weekend', 'rain', 'holiday', 'event']
        target = 'bus_stop_1'
        
        # Ensure all columns exist
        X = train_data[features]
        y = train_data[target]
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train Random Forest Regressor
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train_scaled, y_train)
        
        # Predict on test data
        y_pred = model.predict(X_test_scaled)
        
        # Evaluate model
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print("\n--- Model Performance ---")
        print(f"Mean Squared Error: {mse:.4f}")
        print(f"Mean Absolute Error: {mae:.4f}")
        print(f"R-squared Score: {r2:.4f}")
        
        # Create next day features
        next_day_features = create_next_day_features(train_data)
        
        # Scale next day features
        next_day_features_scaled = scaler.transform(next_day_features)
        
        # Predict for next day
        next_day_predictions = model.predict(next_day_features_scaled)
        
        # Print next day predictions in a formatted manner
        print("\n--- Next Day Predictions ---")
        print("Hour | Predicted Bus Stop 1 Passengers")
        print("-" * 40)
        for hour, prediction in zip(next_day_features['hour'], next_day_predictions):
            print(f"{hour:4d} | {prediction:6.2f}")
        
        return model, scaler, next_day_predictions
    
    except KeyError as e:
        print(f"Column error: {e}")
        print("Available columns:", train_data.columns)
        return None, None, None
    
    except Exception as e:
        print(f"Unexpected error: {e}")
        return None, None, None

def main():
    # File path - replace with your actual file path

    
    # Load data
    data = load_data(file_path)
    
    if data is not None:
        # Preprocess data
        processed_data = preprocess_data(data)
        
        # Train and predict
        model, scaler, next_day_predictions = train_and_predict(processed_data)
        
        if model and scaler:
            print("\nModel training completed successfully!")

if __name__ == "__main__":
    main()