import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
import os

# Set up the file path
dirname = os.path.dirname(__file__)
dataFilename = os.path.join(dirname, 'boarders2.csv')
testFilename = os.path.join(dirname, 'boarders2test.csv')

busCapacity = 30

def train_and_predict():
    # Load training data
    train_file_path = dataFilename
    train_data = pd.read_csv(train_file_path)
    
    # Feature selection: X will be the features, y will be the target variable (bus adjustments)
    X = train_data[['Day', 'Hour']]  # Features
    y = train_data['Boarders']  # Target variable

    # One-hot encode 'DayOfWeek' and 'TimeOfDay' for model compatibility
    X = pd.get_dummies(X, columns=['Day', 'Hour'])

    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Handle missing values by imputing them with the mean of the column
    imputer = SimpleImputer(strategy='mean')
    X_train = imputer.fit_transform(X_train)
    X_test = imputer.transform(X_test)

    # Initialize and train the RandomForestRegressor model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Save the model and column structure for future use
    joblib.dump(model, 'bus_model.pkl')
    joblib.dump(X.columns, 'model_columns.pkl')

    # Print model performance on test data
    score = model.score(X_test, y_test)
    print(f"Model trained. Test score: {score:.2f}")

def predict_on_new_data():
    
    model = joblib.load('bus_model.pkl')
    model_columns = joblib.load('model_columns.pkl')

    test_file_path = testFilename
    test_data = pd.read_csv(test_file_path)

    try:
        X_new = test_data[['Day', 'Hour', 'Boarders']]

        X_new = pd.get_dummies(X_new, columns=['Day', 'Hour'])

        # Align columns with the model_columns (ensure they match)
        X_new = X_new.reindex(columns=model_columns, fill_value=0)

        # Handle missing values (fill NaNs with 0 or other suitable strategy)
        imputer = SimpleImputer(strategy='mean')  # You can also use 'most_frequent', 'median', etc.
        X_new = imputer.fit_transform(X_new)  # Apply imputation

        # Predict bus adjustments (i.e., how many more/less buses to deploy)
        predictions = model.predict(X_new)  # Pass as NumPy array

        # Interpret the prediction
        predicted_no = predictions  # Assuming you're predicting for just one row
        for i in range(len(predictions)):
            print("prediction:",i , (predictions[i]//1) + 1 )
            print("Bus count: ", ((predictions[i]//1) + 1)//busCapacity +1)

        # Check if the prediction is positive or negative
        '''
        if predicted_buses > 0:
            print(f"Recommendation: Release {round(predicted_buses)} more buses.")
        elif predicted_buses < 0:
            print(f"Recommendation: Reduce the number of buses by {round(abs(predicted_buses))}.")
        else:
            print("No change in the number of buses is recommended.")
            

        # Estimate the bus capacity (average number of people per bus)
        average_people_per_bus = 30  # Example: estimated average of 30 people per bus

        # For example, say the total number of people at the stops is 600 for this prediction scenario
        total_people_at_stops = test_data[['A', 'B', 'C', 'D', 'E']].sum().sum()  # Sum of people at all stops

        # Estimate how many buses would be required based on people per bus
        estimated_buses_needed = total_people_at_stops / average_people_per_bus

        print(f"Estimated buses needed based on people count: {round(estimated_buses_needed)} buses")
        print("------")
        '''

    except KeyError as e:
        print(f"Column not found in test data: {e}")
    except ValueError as e:
        print(f"Value error: {e}")

# Run training
train_and_predict()

# Run prediction on new data (testing)
predict_on_new_data()
