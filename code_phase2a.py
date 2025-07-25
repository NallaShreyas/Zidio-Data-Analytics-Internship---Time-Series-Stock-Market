import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings
warnings.filterwarnings("ignore")  # Suppress convergence warnings

# Load cleaned data
df = pd.read_csv('cleaned_stock_data.csv')
df['Date'] = pd.to_datetime(df['Date'])

# Define stocks to model (start with RELIANCE, HDFCBANK)
stocks = ['ADANIPORTS', 'ASIANPAINT', 'AXISBANK', 'BAJAJFINSV', 'BAJFINANCE', 'BHARTIARTL', 
          'BPCL', 'BRITANNIA', 'CIPLA', 'COALINDIA', 'DRREDDY', 'EICHERMOT', 'GAIL', 
          'GRASIM', 'HCLTECH', 'HDFC', 'HDFCBANK', 'HEROMOTOCO', 'HINDALCO', 'HINDUNILVR', 
          'ICICIBANK', 'INDUSINDBK', 'INFY', 'IOC', 'ITC', 'JSWSTEEL', 'KOTAKBANK', 'LT', 
          'MARUTI', 'MM', 'NESTLEIND', 'NTPC', 'ONGC', 'POWERGRID', 'SBIN', 'SHREECEM', 
          'SUNPHARMA', 'TATAMOTORS', 'TATASTEEL', 'TCS', 'TECHM', 'TITAN', 'ULTRACEMCO', 
          'UPL', 'VEDL', 'WIPRO', 'ZEEL']
test_size = 126  # ~6 months (Nov 2020–Apr 2021)

# Initialize results
forecasts = []
metrics = []

for stock in stocks:
    # Filter stock data and remove duplicates
    stock_data = df[df['Symbol'] == stock][['Date', 'Volatility']].dropna()
    stock_data = stock_data.drop_duplicates(subset=['Date'])
    if len(stock_data) < 166:  # Need enough for train + test
        print(f"Skipping {stock}: insufficient data ({len(stock_data)} rows).")
        continue
    
    # Split train/test
    train = stock_data[stock_data['Date'] < '2020-11-01']['Volatility']
    test = stock_data[stock_data['Date'] >= '2020-11-01']['Volatility']
    test_dates = stock_data[stock_data['Date'] >= '2020-11-01']['Date']
    
    # Set test size dynamically
    test_size = len(test)
    if test_size == 0:
        print(f"Skipping {stock}: no test data (Nov 2020–Apr 2021).")
        continue
    
    # ARIMA(1,0,1)
    try:
        arima_model = ARIMA(train, order=(1, 0, 1)).fit()
        arima_forecast = arima_model.forecast(steps=test_size)
        arima_aic = arima_model.aic
        arima_sbic = arima_model.bic
    except Exception as e:
        print(f"ARIMA failed for {stock}: {e}")
        arima_forecast = np.full(test_size, np.nan)
        arima_aic, arima_sbic = np.nan, np.nan
    
    # SARIMA(1,0,1)(1,0,1,12)
    try:
        sarima_model = SARIMAX(train, order=(1, 0, 1), seasonal_order=(1, 0, 1, 12)).fit(disp=False)
        sarima_forecast = sarima_model.forecast(steps=test_size)
        sarima_aic = sarima_model.aic
        sarima_sbic = sarima_model.bic
    except Exception as e:
        print(f"SARIMA failed for {stock}: {e}")
        sarima_forecast = np.full(test_size, np.nan)
        sarima_aic, sarima_sbic = np.nan, np.nan
    
    # Debug lengths
    print(f"{stock}: test_dates={len(test_dates)}, test={len(test)}, arima_forecast={len(arima_forecast)}, sarima_forecast={len(sarima_forecast)}")
    
    # Ensure lengths match
    if not (len(test_dates) == len(test) == len(arima_forecast) == len(sarima_forecast)):
        print(f"Length mismatch for {stock}. Skipping forecast DataFrame.")
        continue
    
    # Store forecasts
    forecast_df = pd.DataFrame({
        'Date': test_dates.values,
        'Symbol': stock,
        'Actual': test.values,
        'ARIMA_Forecast': arima_forecast,
        'SARIMA_Forecast': sarima_forecast
    })
    forecasts.append(forecast_df)
    
    # Store metrics
    metrics.append({
        'Symbol': stock,
        'ARIMA_AIC': arima_aic,
        'ARIMA_SBIC': arima_sbic,
        'SARIMA_AIC': sarima_aic,
        'SARIMA_SBIC': sarima_sbic
    })
    
    # Plot forecasts
    plt.figure(figsize=(10, 6))
    plt.plot(stock_data['Date'], stock_data['Volatility'], label='Actual', alpha=0.5)
    plt.plot(test_dates, arima_forecast, label='ARIMA Forecast', linestyle='--')
    plt.plot(test_dates, sarima_forecast, label='SARIMA Forecast', linestyle='-.')
    plt.title(f'Volatility Forecast ({stock})')
    plt.xlabel('Date')
    plt.ylabel('Volatility (21-day Std Dev)')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'forecast_{stock}.png')
    plt.close()

# Save results
if forecasts:
    forecasts_df = pd.concat(forecasts, ignore_index=True)
    forecasts_df.to_csv('phase2a_forecasts.csv', index=False)
else:
    print("No forecasts generated due to errors.")
metrics_df = pd.DataFrame(metrics)
metrics_df.to_csv('phase2a_metrics.csv', index=False)

print("Phase 2a completed. Outputs saved: phase2a_forecasts.csv, phase2a_metrics.csv, forecast_RELIANCE.png, forecast_HDFCBANK.png")

