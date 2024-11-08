'''
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from datetime import datetime
import numpy as np

import os

# Set up the file path
dirname = os.path.dirname(__file__)
filename = os.path.join(dirname, 'PeopleInTheBusStop.csv')

# Load data
df = pd.read_csv(filename)

# Convert 'Date' and 'Time' to a single datetime column
df['Datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])

# Set 'Datetime' as the index
df.set_index('Datetime', inplace=True)

# Drop 'Date', 'Time', and 'DayOfWeek' columns as they're not needed for prediction
df = df.drop(columns=['Date', 'Time', 'DayOfWeek'])

# Visualize the data
df['NoOfPeople'].plot(figsize=(10, 6))
plt.title('Number of People at Bus Stop')
plt.xlabel('Date')
plt.ylabel('No. of People')
plt.show()

# Split the data into training and test sets (80% training, 20% testing)
train_size = int(len(df) * 0.8)
train, test = df[:train_size], df[train_size:]

# Build ARIMA model
# p=1, d=1, q=1 are initial guesses; we'll use Auto ARIMA later to find optimal params
model = ARIMA(train, order=(1, 1, 1))
model_fit = model.fit()

# Make predictions
predictions = model_fit.forecast(steps=len(test))

# Convert predictions into a DataFrame for comparison
test['Predicted_NoOfPeople'] = predictions

# Plot actual vs predicted values
plt.figure(figsize=(10, 6))
plt.plot(test.index, test['NoOfPeople'], label='Actual')
plt.plot(test.index, test['Predicted_NoOfPeople'], label='Predicted', linestyle='--')
plt.legend()
plt.title('ARIMA Model: Actual vs Predicted')
plt.xlabel('Date')
plt.ylabel('No. of People')
plt.show()

# Evaluate the model using RMSE (Root Mean Squared Error)
from sklearn.metrics import mean_squared_error
rmse = np.sqrt(mean_squared_error(test['NoOfPeople'], predictions))
print(f'RMSE: {rmse}')

# Forecasting future values
future_steps = 24  # Forecast the next 24 hours
forecast = model_fit.forecast(steps=future_steps)
forecast_index = pd.date_range(df.index[-1] + pd.Timedelta(hours=1), periods=future_steps, freq='H')

# Create a DataFrame for the forecast
forecast_df = pd.DataFrame(forecast, index=forecast_index, columns=['Forecasted_NoOfPeople'])

# Plot future predictions
plt.figure(figsize=(10, 6))
plt.plot(df.index, df['NoOfPeople'], label='Historical Data')
plt.plot(forecast_df.index, forecast_df['Forecasted_NoOfPeople'], label='Forecast', linestyle='--')
plt.legend()
plt.title('Bus Stop Forecast: Next 24 Hours')
plt.xlabel('Date')
plt.ylabel('No. of People')
plt.show()
'''

import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pmdarima import auto_arima
from sklearn.metrics import mean_squared_error
from datetime import datetime, timedelta
import numpy as np
import os

# Set up the file path
dirname = os.path.dirname(__file__)
filename = os.path.join(dirname, 'PeopleInTheBusStop.csv')

# Load data
df = pd.read_csv(filename)

# Convert 'Date' and 'Time' to a single datetime column
df['Datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])

# Set 'Datetime' as the index
df.set_index('Datetime', inplace=True)

# Drop unnecessary columns
df = df.drop(columns=['Date', 'Time', 'DayOfWeek'])

# Visualize the data
plt.figure(figsize=(10, 6))
plt.plot(df.index, df['NoOfPeople'], label='Historical Data')
plt.title('Number of People at Bus Stop')
plt.xlabel('Date')
plt.ylabel('No. of People')
plt.show()

# Split the data into training and test sets (80% training, 20% testing)
train_size = int(len(df) * 0.8)
train, test = df[:train_size], df[train_size:]

# Automatically find optimal ARIMA parameters with Auto ARIMA
auto_model = auto_arima(train, seasonal=True, m=24,  # m=24 if data has daily seasonality (24-hour cycle)
                        trace=True, error_action='ignore', suppress_warnings=True, stepwise=True)

# Train the SARIMA model with identified parameters
model = SARIMAX(train, order=auto_model.order, seasonal_order=auto_model.seasonal_order)
model_fit = model.fit(disp=False)

# Make predictions on the test set
predictions = model_fit.get_forecast(steps=len(test)).predicted_mean
test['Predicted_NoOfPeople'] = predictions

# Plot actual vs. predicted values
plt.figure(figsize=(10, 6))
plt.plot(test.index, test['NoOfPeople'], label='Actual')
plt.plot(test.index, test['Predicted_NoOfPeople'], label='Predicted', linestyle='--')
plt.legend()
plt.title('SARIMA Model: Actual vs Predicted')
plt.xlabel('Date')
plt.ylabel('No. of People')
plt.show()

# Evaluate the model using RMSE
rmse = np.sqrt(mean_squared_error(test['NoOfPeople'], predictions))
print(f'RMSE: {rmse}')

# Forecast future values
future_steps = 24  # Forecast the next 24 hours
forecast = model_fit.get_forecast(steps=future_steps)
forecast_index = pd.date_range(df.index[-1] + timedelta(hours=1), periods=future_steps, freq='H')
forecast_df = pd.DataFrame(forecast.predicted_mean, index=forecast_index, columns=['Forecasted_NoOfPeople'])

# Plot future predictions
plt.figure(figsize=(10, 6))
plt.plot(df.index, df['NoOfPeople'], label='Historical Data')
plt.plot(forecast_df.index, forecast_df['Forecasted_NoOfPeople'], label='Forecast', linestyle='--')
plt.legend()
plt.title('Bus Stop Forecast: Next 24 Hours')
plt.xlabel('Date')
plt.ylabel('No. of People')
plt.show()
