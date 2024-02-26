from prophet import Prophet
import pandas as pd
import plotly.graph_objects as go
import zero_true as zt


df = pd.read_csv('FCStocks.csv')

df_prophet = df[['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 'y'})

df_prophet['ds'] = pd.to_datetime(df_prophet['ds'])

m = Prophet()

m.fit(df_prophet)

future = m.make_future_dataframe(periods=365)

forecast = m.predict(future)

layout = {
    "title": "Forecasted Stock Prices",
    "xaxis": {"title": "Date"},
    "yaxis": {"title": "Stock Price"}
}

fig = go.Figure()

fig.add_trace(go.Scatter(x=df_prophet['ds'], y=df_prophet['y'], name='Historical Close'))

fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], name='Forecast'))

fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_upper'], name='Upper Confidence Interval', line=dict(width=0)))
fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_lower'], name='Lower Confidence Interval', line=dict(width=0), fill='tonexty'))

fig.update_layout(layout)

fig.show()
