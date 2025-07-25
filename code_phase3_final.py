import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import StandardScaler
import os
from datetime import timedelta

# Load data
data = pd.read_csv('cleaned_stock_data.csv')
data['Date'] = pd.to_datetime(data['Date'])
best_models = pd.read_csv('phase2c_best_models.csv')
metrics_2b = pd.read_csv('phase2b_metrics.csv')

# Define stocks
stocks = ['ADANIPORTS', 'AXISBANK', 'BAJAJFINSV', 'BAJFINANCE', 'BHARTIARTL', 'BPCL', 
          'BRITANNIA', 'CIPLA', 'COALINDIA', 'DRREDDY', 'EICHERMOT', 'GAIL', 'GRASIM', 
          'HCLTECH', 'HDFC', 'HDFCBANK', 'HINDUNILVR', 'ICICIBANK', 'INFY', 'ITC', 
          'JSWSTEEL', 'KOTAKBANK', 'LT', 'MARUTI', 'NESTLEIND', 'POWERGRID', 'SBIN', 
          'SHREECEM', 'TATAMOTORS', 'TCS', 'TECHM', 'VEDL', 'WIPRO', 'ZEEL']

# Create directory for plots
os.makedirs('recommendation_plots', exist_ok=True)

# Forecast parameters
forecast_horizon = 126  # ~6 months (May 2021–Oct 2021)
last_date = data['Date'].max()
forecast_dates = pd.date_range(start=last_date + timedelta(days=1), periods=forecast_horizon, freq='B')

# Initialize results
forecasts = []
recommendations = []

# Compute historical Volatility median
volatility_median = data.groupby('Symbol')['Volatility'].median().to_dict()

for stock in stocks:
    # Filter stock data
    stock_data = data[data['Symbol'] == stock][['Date', 'Volatility', 'Volume', 'Turnover', '%Deliverable', 'Close']].dropna()
    stock_data = stock_data.drop_duplicates(subset=['Date'])
    if len(stock_data) < 166:
        print(f"Skipping {stock}: insufficient data ({len(stock_data)} rows).")
        continue
    
    # Get best model and check Durbin-Watson
    best_model_row = best_models[best_models['Symbol'] == stock]
    if best_model_row.empty:
        print(f"Skipping {stock}: no best model.")
        continue
    best_model = best_model_row['Best_Model'].iloc[0]
    
    # Check for serial correlation
    dw = metrics_2b[metrics_2b['Symbol'] == stock]['Durbin_Watson']
    use_arima = False
    if dw.empty:
        print(f"Warning: No Durbin-Watson for {stock}, using {best_model}.")
    elif dw.iloc[0] < 1.5:
        print(f"Switching to ARIMA for {stock}: Durbin-Watson {dw.iloc[0]:.2f} < 1.5.")
        use_arima = True
    
    # Prepare features
    stock_data['Volatility_Lag1'] = stock_data['Volatility'].shift(1)
    stock_data['Log_Volume'] = np.log1p(stock_data['Volume'])
    stock_data['Log_Turnover'] = np.log1p(stock_data['Turnover'])
    stock_data = stock_data.dropna()
    
    # Compute 21-day SMA slope for April 2021
    last_month = stock_data[(stock_data['Date'].dt.month == 4) & (stock_data['Date'].dt.year == 2021)]
    sma_slope = np.nan
    if len(last_month) >= 21:
        last_month = last_month.sort_values('Date')
        sma = last_month['Close'].rolling(window=21, min_periods=21).mean()
        sma_diff = sma.diff()
        sma_slope = sma_diff.iloc[-1] / sma.iloc[-1] if not sma.empty else 0  # Normalized slope
        if np.isnan(sma_slope):
            sma_slope = 0  # Default to neutral
    
    # Forecast Volatility
    if best_model == 'Regression' and not use_arima:
        # Get regression coefficients
        coeffs = metrics_2b[metrics_2b['Symbol'] == stock][['Coeff_Intercept', 'Coeff_Log_Volume', 'Coeff_Log_Turnover', 
                                                           'Coeff_%Deliverable', 'Coeff_Volatility_Lag1']]
        if coeffs.empty:
            print(f"Skipping {stock}: no regression coefficients.")
            continue
        coeffs = coeffs.iloc[0]
        
        # Use Apr 2021 averages for predictors
        if last_month.empty:
            print(f"Skipping {stock}: no Apr 2021 data.")
            continue
        predictors = {
            'Log_Volume': last_month['Log_Volume'].mean(),
            'Log_Turnover': last_month['Log_Turnover'].mean(),
            '%Deliverable': last_month['%Deliverable'].mean(),
            'Volatility_Lag1': stock_data['Volatility'].iloc[-1]
        }
        
        # Standardize predictors
        train = stock_data[stock_data['Date'] < '2020-11-01']
        scaler = StandardScaler()
        scaler.fit(train[['Log_Volume', 'Log_Turnover', '%Deliverable', 'Volatility_Lag1']])
        X_pred = scaler.transform(pd.DataFrame([predictors], columns=['Log_Volume', 'Log_Turnover', '%Deliverable', 'Volatility_Lag1']))
        
        # Forecast
        forecast = coeffs['Coeff_Intercept'] + np.sum(X_pred[0] * coeffs[1:])
        forecast_values = np.full(forecast_horizon, forecast)
    
    else:  # ARIMA (either best model or fallback)
        # Set Date as index for ARIMA
        stock_data_arima = stock_data.set_index('Date')['Volatility']
        # Fit ARIMA(1,0,1)
        try:
            model = ARIMA(stock_data_arima, order=(1, 0, 1)).fit()
            forecast_values = model.forecast(steps=forecast_horizon)
        except Exception as e:
            print(f"ARIMA failed for {stock}: {e}")
            forecast_values = np.full(forecast_horizon, np.nan)
    
    # Store forecasts
    forecast_df = pd.DataFrame({
        'Date': forecast_dates,
        'Symbol': stock,
        'Forecast_Volatility': forecast_values
    })
    forecasts.append(forecast_df)
    
    # Generate recommendations
    median_vol = volatility_median.get(stock, np.median(stock_data['Volatility']))
    avg_forecast_vol = np.nanmean(forecast_values)
    
    # Refined thresholds
    risk_averse_vol = 'Buy' if avg_forecast_vol < median_vol * 1.0 else 'Sell' if avg_forecast_vol > median_vol * 1.2 else 'Hold'
    risk_neutral_vol = 'Buy' if avg_forecast_vol < median_vol * 0.9 else 'Sell' if avg_forecast_vol > median_vol * 1.4 else 'Hold'
    risk_tolerant_vol = 'Buy' if avg_forecast_vol > median_vol else 'Sell' if avg_forecast_vol > median_vol * 2.0 else 'Hold'
    
    # Incorporate price trend (SMA slope)
    risk_averse = risk_averse_vol if sma_slope > -0.001 or risk_averse_vol != 'Buy' else 'Hold'  # Allow slightly negative slope
    risk_neutral = risk_neutral_vol if risk_neutral_vol != 'Hold' else ('Buy' if sma_slope > 0.002 else 'Sell' if sma_slope < -0.002 else 'Hold')
    risk_tolerant = risk_tolerant_vol  # Price trend less critical
    
    recommendations.append({
        'Symbol': stock,
        'Avg_Forecast_Volatility': avg_forecast_vol,
        'Median_Volatility': median_vol,
        'SMA_Slope': sma_slope,
        'Risk_Averse': risk_averse,
        'Risk_Neutral': risk_neutral,
        'Risk_Tolerant': risk_tolerant
    })
    
    # Plot for HDFCBANK and MARUTI
    if stock in ['HDFCBANK', 'MARUTI']:
        plt.figure(figsize=(10, 6))
        plt.plot(stock_data['Date'], stock_data['Volatility'], label='Historical', alpha=0.5)
        plt.plot(forecast_dates, forecast_values, label='Forecast', linestyle='--')
        plt.axhline(median_vol, color='r', linestyle=':', label='Median Volatility')
        plt.title(f'Volatility Forecast ({stock}, Model: {"ARIMA" if use_arima or best_model == "ARIMA" else "Regression"})')
        plt.xlabel('Date')
        plt.ylabel('Volatility (21-day Std Dev)')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'recommendation_plots/forecast_{stock}.png')
        plt.close()

# Save results
if forecasts:
    forecasts_df = pd.concat(forecasts, ignore_index=True)
    forecasts_df.to_csv('phase3_forecasts.csv', index=False)
else:
    print("No forecasts generated.")
recommendations_df = pd.DataFrame(recommendations)
recommendations_df.to_csv('phase3_recommendations.csv', index=False)

# Generate summary report
with open('phase3_summary.txt', 'w') as f:
    f.write("Phase 3: Recommendation System Summary\n")
    f.write("=====================================\n")
    f.write(f"Stocks Evaluated: {len(recommendations)}\n")
    arima_count = sum(1 for r in recommendations if r['Symbol'] in best_models[best_models['Best_Model'] == 'ARIMA']['Symbol'].values or (metrics_2b[metrics_2b['Symbol'] == r['Symbol']]['Durbin_Watson'].iloc[0] < 1.5 if r['Symbol'] in metrics_2b['Symbol'].values else False))
    f.write(f"Models Used: Regression for {len(recommendations) - arima_count}, ARIMA for {arima_count} (including {sum(1 for _, row in metrics_2b.iterrows() if row['Durbin_Watson'] < 1.5 and row['Symbol'] in recommendations_df['Symbol'].values)} due to serial correlation)\n")
    for profile in ['Risk_Averse', 'Risk_Neutral', 'Risk_Tolerant']:
        counts = recommendations_df[profile].value_counts()
        f.write(f"\n{profile} Recommendations:\n")
        for signal, count in counts.items():
            f.write(f"  {signal}: {count} stocks\n")
    f.write("\nKey Observations:\n")
    f.write("- Refined thresholds: 1.0/1.2 for risk-averse, 0.9/1.4 for risk-neutral, 2.0 Sell for risk-tolerant.\n")
    f.write("- Price trends (21-day SMA slope) allow slightly negative slope (-0.001) for risk-averse Buy.\n")
    f.write("- ARIMA used for stocks with Durbin-Watson < 1.5 to address serial correlation.\n")
    f.write("- Check phase3_recommendations.csv for stock-specific signals and SMA slopes.\n")

print("Phase 3 completed. Outputs saved: phase3_forecasts.csv, phase3_recommendations.csv, recommendation_plots/forecast_HDFCBANK.png, recommendation_plots/forecast_MARUTI.png, phase3_summary.txt")