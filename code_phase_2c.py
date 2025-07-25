import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
import os

# Load data
forecasts_2a = pd.read_csv('phase2a_forecasts.csv')
forecasts_2a['Date'] = pd.to_datetime(forecasts_2a['Date'])
predictions_2b = pd.read_csv('phase2b_predictions.csv')
predictions_2b['Date'] = pd.to_datetime(predictions_2b['Date'])
metrics_2a = pd.read_csv('phase2a_metrics.csv')
metrics_2b = pd.read_csv('phase2b_metrics.csv')

# Define stocks
stocks = ['ADANIPORTS', 'AXISBANK', 'BAJAJFINSV', 'BAJFINANCE', 'BHARTIARTL', 'BPCL', 
          'BRITANNIA', 'CIPLA', 'COALINDIA', 'DRREDDY', 'EICHERMOT', 'GAIL', 'GRASIM', 
          'HCLTECH', 'HDFC', 'HDFCBANK', 'HINDUNILVR', 'ICICIBANK', 'INFY', 'ITC', 
          'JSWSTEEL', 'KOTAKBANK', 'LT', 'MARUTI', 'NESTLEIND', 'POWERGRID', 'SBIN', 
          'SHREECEM', 'TATAMOTORS', 'TCS', 'TECHM', 'VEDL', 'WIPRO', 'ZEEL']

# Create directory for plots
os.makedirs('comparison_plots', exist_ok=True)

# Initialize results
eval_metrics = []
best_models = []

for stock in stocks:
    # Filter data
    stock_2a = forecasts_2a[forecasts_2a['Symbol'] == stock]
    stock_2b = predictions_2b[predictions_2b['Symbol'] == stock]
    
    # Merge on Date for consistent test period
    merged = stock_2a.merge(stock_2b[['Date', 'Predicted']], on='Date', how='inner')
    if merged.empty:
        print(f"Skipping {stock}: no overlapping test data.")
        continue
    
    # Extract actual and predicted values
    actual = merged['Actual']
    arima_pred = merged['ARIMA_Forecast']
    sarima_pred = merged['SARIMA_Forecast']
    reg_pred = merged['Predicted']
    test_dates = merged['Date']
    
    # Compute metrics (skip NaNs)
    def compute_metrics(y_true, y_pred):
        mask = ~np.isnan(y_pred)
        if mask.sum() < 2:
            return np.nan, np.nan
        return (
            np.sqrt(mean_squared_error(y_true[mask], y_pred[mask])),
            mean_absolute_error(y_true[mask], y_pred[mask])
        )
    
    arima_rmse, arima_mae = compute_metrics(actual, arima_pred)
    sarima_rmse, sarima_mae = compute_metrics(actual, sarima_pred)
    reg_rmse, reg_mae = compute_metrics(actual, reg_pred)
    
    # Get model fit metrics
    metrics_2a_stock = metrics_2a[metrics_2a['Symbol'] == stock]
    metrics_2b_stock = metrics_2b[metrics_2b['Symbol'] == stock]
    
    arima_aic = metrics_2a_stock['ARIMA_AIC'].iloc[0] if not metrics_2a_stock.empty else np.nan
    arima_sbic = metrics_2a_stock['ARIMA_SBIC'].iloc[0] if not metrics_2a_stock.empty else np.nan
    sarima_aic = metrics_2a_stock['SARIMA_AIC'].iloc[0] if not metrics_2a_stock.empty else np.nan
    sarima_sbic = metrics_2a_stock['SARIMA_SBIC'].iloc[0] if not metrics_2a_stock.empty else np.nan
    
    # Pseudo-AIC for regression (based on RSS)
    if not metrics_2b_stock.empty:
        rss = np.sum((actual - reg_pred)**2)
        n = len(actual)
        k = 5  # 4 predictors + intercept
        reg_aic = n * np.log(rss / n) + 2 * k
        reg_sbic = n * np.log(rss / n) + k * np.log(n)
        r_squared = metrics_2b_stock['R_Squared'].iloc[0]
        durbin_watson = metrics_2b_stock['Durbin_Watson'].iloc[0]
    else:
        reg_aic, reg_sbic, r_squared, durbin_watson = np.nan, np.nan, np.nan, np.nan
    
    # Store evaluation metrics
    eval_metrics.append({
        'Symbol': stock,
        'ARIMA_RMSE': arima_rmse,
        'ARIMA_MAE': arima_mae,
        'ARIMA_AIC': arima_aic,
        'ARIMA_SBIC': arima_sbic,
        'SARIMA_RMSE': sarima_rmse,
        'SARIMA_MAE': sarima_mae,
        'SARIMA_AIC': sarima_aic,
        'SARIMA_SBIC': sarima_sbic,
        'Regression_RMSE': reg_rmse,
        'Regression_MAE': reg_mae,
        'Regression_AIC': reg_aic,
        'Regression_SBIC': reg_sbic,
        'Regression_R_Squared': r_squared,
        'Regression_Durbin_Watson': durbin_watson
    })
    
    # Select best model (lowest RMSE)
    rmse_values = {
        'ARIMA': arima_rmse,
        'SARIMA': sarima_rmse,
        'Regression': reg_rmse
    }
    if all(np.isnan(list(rmse_values.values()))):
        best_model = 'None'
    else:
        best_model = min(rmse_values, key=lambda k: rmse_values[k] if not np.isnan(rmse_values[k]) else float('inf'))
    best_models.append({'Symbol': stock, 'Best_Model': best_model, 'Best_RMSE': rmse_values[best_model]})
    
    # Plot for HDFCBANK and MARUTI
    if stock in ['ADANIPORTS', 'AXISBANK', 'BAJAJFINSV', 'BAJFINANCE', 'BHARTIARTL', 'BPCL', 
          'BRITANNIA', 'CIPLA', 'COALINDIA', 'DRREDDY', 'EICHERMOT', 'GAIL', 'GRASIM', 
          'HCLTECH', 'HDFC', 'HDFCBANK', 'HINDUNILVR', 'ICICIBANK', 'INFY', 'ITC', 
          'JSWSTEEL', 'KOTAKBANK', 'LT', 'MARUTI', 'NESTLEIND', 'POWERGRID', 'SBIN', 
          'SHREECEM', 'TATAMOTORS', 'TCS', 'TECHM', 'VEDL', 'WIPRO', 'ZEEL']:
        plt.figure(figsize=(12, 6))
        plt.plot(test_dates, actual, label='Actual', alpha=0.5)
        plt.plot(test_dates, arima_pred, label='ARIMA', linestyle='--')
        plt.plot(test_dates, sarima_pred, label='SARIMA', linestyle='-.')
        plt.plot(test_dates, reg_pred, label='Regression', linestyle=':')
        plt.title(f'Volatility Forecast Comparison ({stock})')
        plt.xlabel('Date')
        plt.ylabel('Volatility (21-day Std Dev)')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'comparison_plots/comparison_{stock}.png')
        plt.close()

# Save results
eval_metrics_df = pd.DataFrame(eval_metrics)
eval_metrics_df.to_csv('phase2c_eval_metrics.csv', index=False)
best_models_df = pd.DataFrame(best_models)
best_models_df.to_csv('phase2c_best_models.csv', index=False)

# Generate summary report
with open('phase2c_summary.txt', 'w') as f:
    f.write("Phase 2c: Model Comparison Summary\n")
    f.write("================================\n")
    f.write(f"Stocks Evaluated: {len(eval_metrics)}\n")
    best_model_counts = best_models_df['Best_Model'].value_counts()
    for model, count in best_model_counts.items():
        f.write(f"{model} selected as best for {count} stocks\n")
    f.write("\nKey Observations:\n")
    f.write("- ARIMA/SARIMA: Better for stocks with strong autocorrelation (Phase 1 ACF).\n")
    f.write("- Regression: Better when liquidity metrics (Volume, Turnover) drive Volatility.\n")
    f.write("- Check Durbin-Watson < 1.5 in phase2c_eval_metrics.csv for regression serial correlation.\n")
    f.write("- Stocks with high RMSE may need model refinement (e.g., different ARIMA orders).\n")

print("Phase 2c completed. Outputs saved: phase2c_eval_metrics.csv, phase2c_best_models.csv, comparison_plots/comparison_HDFCBANK.png, comparison_plots/comparison_MARUTI.png, phase2c_summary.txt")