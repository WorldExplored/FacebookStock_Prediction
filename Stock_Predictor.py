

#import zero_true as zt
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from prophet import Prophet

# Load your dataset
df = pd.read_csv('FCStocks.csv')

# Prepare the dataset for Prophet
df_prophet = df[['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 'y'})

# Convert 'ds' to datetime if it's not already
df_prophet['ds'] = pd.to_datetime(df_prophet['ds'])


# Create a Prophet model instance
m = Prophet()

# Fit the model with your dataframe
m.fit(df_prophet)


# Create future dataframe for 365 days into the future
future = m.make_future_dataframe(periods=365)

# Predict future stock prices
forecast = m.predict(future)


# Inspect the last few predictions
print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())

# Plot the forecast
fig1 = m.plot(forecast)

# Plot forecast components
fig2 = m.plot_components(forecast)
