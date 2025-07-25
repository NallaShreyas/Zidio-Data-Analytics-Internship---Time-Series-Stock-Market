import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.stattools import durbin_watson
import os

# Load cleaned data
df = pd.read_csv('cleaned_stock_data.csv')
df['Date'] = pd.to_datetime(df['Date'])

# Define stocks
stocks = ['ADANIPORTS', 'AXISBANK', 'BAJAJFINSV', 'BAJFINANCE', 'BHARTIARTL', 'BPCL', 
          'BRITANNIA', 'CIPLA', 'COALINDIA', 'DRREDDY', 'EICHERMOT', 'GAIL', 'GRASIM', 
          'HCLTECH', 'HDFC', 'HDFCBANK', 'HINDUNILVR', 'ICICIBANK', 'INFY', 'ITC', 
          'JSWSTEEL', 'KOTAKBANK', 'LT', 'MARUTI', 'NESTLEIND', 'POWERGRID', 'SBIN', 
          'SHREECEM', 'TATAMOTORS', 'TCS', 'TECHM', 'VEDL', 'WIPRO', 'ZEEL']

# Create directory for plots
os.makedirs('regression_plots', exist_ok=True)

# Initialize results
predictions = []
metrics = []

# Feature engineering
for stock in stocks:
    # Filter stock data
    stock_data = df[df['Symbol'] == stock][['Date', 'Volatility', 'Volume', 'Turnover', '%Deliverable']].dropna()
    stock_data = stock_data.drop_duplicates(subset=['Date'])
    if len(stock_data) < 166:  # Need enough for train + test
        print(f"Skipping {stock}: insufficient data ({len(stock_data)} rows).")
        continue
    
    # Add lagged Volatility
    stock_data['Volatility_Lag1'] = stock_data['Volatility'].shift(1)
    stock_data = stock_data.dropna()  # Remove rows with NaN lag
    
    # Log-transform Volume and Turnover
    stock_data['Log_Volume'] = np.log1p(stock_data['Volume'])
    stock_data['Log_Turnover'] = np.log1p(stock_data['Turnover'])
    
    # Split train/test
    train = stock_data[stock_data['Date'] < '2020-11-01']
    test = stock_data[stock_data['Date'] >= '2020-11-01']
    if len(test) == 0:
        print(f"Skipping {stock}: no test data.")
        continue
    
    # Features and target
    features = ['Log_Volume', 'Log_Turnover', '%Deliverable', 'Volatility_Lag1']
    X_train = train[features]
    y_train = train['Volatility']
    X_test = test[features]
    y_test = test['Volatility']
    test_dates = test['Date']
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Add constant for statsmodels
    X_train_sm = sm.add_constant(X_train_scaled)
    X_test_sm = sm.add_constant(X_test_scaled)
    
    # Fit regression model
    try:
        model = sm.OLS(y_train, X_train_sm).fit()
        y_pred = model.predict(X_test_sm)
        
        # Metrics
        r_squared = model.rsquared
        adj_r_squared = model.rsquared_adj
        dw_stat = durbin_watson(model.resid)
        coeffs = dict(zip(['Intercept'] + features, model.params))
        
        # Store predictions
        pred_df = pd.DataFrame({
            'Date': test_dates,
            'Symbol': stock,
            'Actual': y_test,
            'Predicted': y_pred
        })
        predictions.append(pred_df)
        
        # Store metrics
        metrics.append({
            'Symbol': stock,
            'R_Squared': r_squared,
            'Adj_R_Squared': adj_r_squared,
            'Durbin_Watson': dw_stat,
            **{f'Coeff_{k}': v for k, v in coeffs.items()}
        })
        
        # Plot for HDFCBANK and MARUTI
        if stock in ['ADANIPORTS', 'AXISBANK', 'BAJAJFINSV', 'BAJFINANCE', 'BHARTIARTL', 'BPCL', 
          'BRITANNIA', 'CIPLA', 'COALINDIA', 'DRREDDY', 'EICHERMOT', 'GAIL', 'GRASIM', 
          'HCLTECH', 'HDFC', 'HDFCBANK', 'HINDUNILVR', 'ICICIBANK', 'INFY', 'ITC', 
          'JSWSTEEL', 'KOTAKBANK', 'LT', 'MARUTI', 'NESTLEIND', 'POWERGRID', 'SBIN', 
          'SHREECEM', 'TATAMOTORS', 'TCS', 'TECHM', 'VEDL', 'WIPRO', 'ZEEL']:
            plt.figure(figsize=(10, 6))
            plt.plot(stock_data['Date'], stock_data['Volatility'], label='Actual', alpha=0.5)
            plt.plot(test_dates, y_pred, label='Predicted', linestyle='--')
            plt.title(f'Regression Volatility Prediction ({stock})')
            plt.xlabel('Date')
            plt.ylabel('Volatility (21-day Std Dev)')
            plt.legend()
            plt.grid(True)
            plt.savefig(f'regression_plots/regression_{stock}.png')
            plt.close()
    
    except Exception as e:
        print(f"Regression failed for {stock}: {e}")
        continue

# Save results
if predictions:
    predictions_df = pd.concat(predictions, ignore_index=True)
    predictions_df.to_csv('phase2b_predictions.csv', index=False)
else:
    print("No predictions generated due to errors.")
metrics_df = pd.DataFrame(metrics)
metrics_df.to_csv('phase2b_metrics.csv', index=False)

print("Phase 2b completed. Outputs saved: phase2b_predictions.csv, phase2b_metrics.csv, regression_plots/regression_HDFCBANK.png, regression_plots/regression_MARUTI.png")