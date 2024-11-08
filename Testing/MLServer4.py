import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.impute import SimpleImputer
import os

# Set up the file path
dirname = os.path.dirname(__file__)
dataFilename = os.path.join(dirname, 'boarders2.csv')
testFilename = os.path.join(dirname, 'boarders2test.csv')

busCapacity = 30

def findNoOfBuses(no, max):
    return ((no // max)+1)



def train_and_predict():
    # Load training data
    train_file_path = dataFilename
    train_data = pd.read_csv(train_file_path)

    # Feature engineering: Create additional features

    # Feature selection: X will be the features, y will be the target variable (bus adjustments)
    X = train_data[['Day', 'Hour', 'is_weekend']]  # Features
    y = train_data['Boarders']  # Target variable

    # One-hot encode 'Day' and 'Hour' for model compatibility
    X = pd.get_dummies(X, columns=['Day', 'Hour'])

    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Handle missing values by imputing them with the mean of the column
    imputer = SimpleImputer(strategy='mean')
    X_train = imputer.fit_transform(X_train)
    X_test = imputer.transform(X_test)

    # Initialize and train the RandomForestRegressor model with hyperparameter tuning
    model = RandomForestRegressor(random_state=42)

    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
    }

    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3)
    grid_search.fit(X_train, y_train)
    model = grid_search.best_estimator_

    # Save the model and column structure for future use
    joblib.dump(model, 'bus_model.pkl')
    joblib.dump(X.columns, 'model_columns.pkl')

    # Print model performance on test data
    score = model.score(X_test, y_test)
    print(f"Model trained. Test score: {score:.2f}")

    # Cross-validation score
    cv_scores = cross_val_score(model, X_train, y_train, cv=5)
    print(f"Cross-Validation Scores: {cv_scores}")
    print(f"Mean CV Score: {cv_scores.mean():.2f}")

def predict_on_new_data():
    model = joblib.load('bus_model.pkl')
    model_columns = joblib.load('model_columns.pkl')

    test_file_path = testFilename
    test_data = pd.read_csv(test_file_path)

    try:
        # Feature engineering for test data

        X_new = test_data[['Day', 'Hour', 'is_weekend', 'Boarders']]
        X_new = pd.get_dummies(X_new, columns=['Day', 'Hour'])

        # Alignment
        X_new = X_new.reindex(columns=model_columns, fill_value=0)

        # Handle missing values (fill NaNs with mean)
        imputer = SimpleImputer(strategy='mean')
        X_new = imputer.fit_transform(X_new)

        # Predict bus adjustments (i.e., how many more/less buses to deploy)
        predictions = model.predict(X_new)

        # Interpret the prediction
        for i in range(len(predictions)):
            print("Prediction:", i, (predictions[i] // 1) + 1 ,"\tBus count:", ((predictions[i] // 1) + 1) // busCapacity + 1)

    except KeyError as e:
        print(f"Column not found in test data: {e}")
    except ValueError as e:
        print(f"Value error: {e}")

# Run training
train_and_predict()

# Run prediction on new data (testing)
predict_on_new_data()