import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller, acf
import os

# List of company symbols
symbols = ['ADANIPORTS', 'ASIANPAINT', 'AXISBANK', 'BAJAJFINSV', 'BAJFINANCE', 'BHARTIARTL', 
           'BPCL', 'BRITANNIA', 'CIPLA', 'COALINDIA', 'DRREDDY', 'EICHERMOT', 'GAIL', 
           'GRASIM', 'HCLTECH', 'HDFC', 'HDFCBANK', 'HEROMOTOCO', 'HINDALCO', 'HINDUNILVR', 
           'ICICIBANK', 'INDUSINDBK', 'INFY', 'IOC', 'ITC', 'JSWSTEEL', 'KOTAKBANK', 'LT', 
           'MARUTI', 'MM', 'NESTLEIND', 'NTPC', 'ONGC', 'POWERGRID', 'SBIN', 'SHREECEM', 
           'SUNPHARMA', 'TATAMOTORS', 'TATASTEEL', 'TCS', 'TECHM', 'TITAN', 'ULTRACEMCO', 
           'UPL', 'VEDL', 'WIPRO', 'ZEEL']

# Task 1: Load and merge CSV files
dfs = []
for symbol in symbols:
    file_path = f"{symbol}.csv"
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        dfs.append(df)
    else:
        print(f"Warning: {file_path} not found.")
df = pd.concat(dfs, ignore_index=True)

# Task 2: Data Cleaning
df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
df = df[df['Series'] == 'EQ']
price_cols = ['Prev Close', 'Open', 'High', 'Low', 'Last', 'Close', 'VWAP']
df[price_cols] = df[price_cols].fillna(method='ffill')
df[['Volume', 'Turnover', 'Trades', 'Deliverable Volume']] = df[['Volume', 'Turnover', 'Trades', 'Deliverable Volume']].fillna(0)
df['%Deliverable'] = df['%Deliverable'].fillna(0).clip(0, 1)
df = df.drop_duplicates(subset=['Date', 'Symbol'])
df = df.sort_values(['Symbol', 'Date'])

# Task 3: Volatility Calculation
df['Daily_Return'] = df.groupby('Symbol')['Close'].pct_change()
df['Volatility'] = df.groupby('Symbol')['Daily_Return'].rolling(window=21, min_periods=1).std().reset_index(level=0, drop=True)
df['Range_Volatility'] = (df['High'] - df['Low']) / df['Close']

# Task 4: Descriptive Statistics
liquidity_stats = df.groupby('Symbol')[['Volume', 'Turnover', '%Deliverable', 'Volatility']].describe()
liquidity_stats.to_csv('liquidity_stats.csv')

# Task 5: Stationarity and Autocorrelations
def adf_test(series, name):
    series = series.dropna()
    if len(series) < 2:
        return {'ADF Statistic': np.nan, 'p-value': np.nan}
    result = adfuller(series)
    return {'ADF Statistic': result[0], 'p-value': result[1]}

stationarity_vol = df.groupby('Symbol')['Volatility'].apply(lambda x: adf_test(x, 'Volatility'))
stationarity_vol_df = pd.DataFrame(stationarity_vol.tolist(), index=stationarity_vol.index)
stationarity_vol_df.to_csv('stationarity_volatility.csv')

# ACF for Volatility (all companies)
for symbol in symbols:
    vol_series = df[df['Symbol'] == symbol]['Volatility'].dropna()
    if len(vol_series) < 40:
        print(f"Skipping ACF for {symbol}: insufficient data ({len(vol_series)} points).")
        continue
    acf_vals = acf(vol_series, nlags=40, fft=False)
    plt.figure(figsize=(10, 6))
    plt.stem(range(len(acf_vals)), acf_vals)
    plt.title(f'ACF of Volatility ({symbol})')
    plt.xlabel('Lag')
    plt.ylabel('Autocorrelation')
    plt.ylim(-1, 1)
    plt.grid(True)
    plt.savefig(f'acf_{symbol}_volatility.png')
    plt.close()

# Task 6: Correlation Analysis
corr_matrix = df[['Volatility', 'Range_Volatility', 'Volume', 'Turnover', '%Deliverable']].corr()
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Correlation Matrix')
plt.savefig('correlation_heatmap.png')
plt.close()
corr_matrix.to_csv('correlation_matrix.csv')

# Task 7: Visualizations
# Volatility trend plots (5 companies per plot)
symbols_sorted = sorted(symbols)  # Sort for consistency
chunk_size = 5
for i in range(0, len(symbols_sorted), chunk_size):
    chunk = symbols_sorted[i:i + chunk_size]
    plt.figure(figsize=(12, 6))
    for symbol in chunk:
        temp = df[df['Symbol'] == symbol]
        plt.plot(temp['Date'], temp['Volatility'], label=symbol)
    plt.title(f'Volatility Over Time (Stocks {i+1}-{i+len(chunk)})')
    plt.xlabel('Date')
    plt.ylabel('Volatility (21-day Std Dev)')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'volatility_trend_{i//chunk_size + 1}.png')
    plt.close()
    print(f"Saved volatility_trend_{i//chunk_size + 1}.png")

# Liquidity metrics plots (one per company)
for symbol in symbols:
    stock_data = df[df['Symbol'] == symbol]
    if stock_data.empty:
        print(f"Skipping liquidity plot for {symbol}: no data.")
        continue
    plt.figure(figsize=(12, 6))
    plt.plot(stock_data['Date'], stock_data['Volume'], label='Volume')
    plt.plot(stock_data['Date'], stock_data['Turnover'] / 1e9, label='Turnover (Billions)')
    plt.plot(stock_data['Date'], stock_data['%Deliverable'] * 100, label='%Deliverable')
    plt.title(f'Liquidity Metrics Over Time ({symbol})')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'liquidity_{symbol}.png')
    plt.close()
    print(f"Saved liquidity_{symbol}.png")

# Distribution of liquidity metrics
plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
sns.histplot(df['Volume'], bins=50, log_scale=True)
plt.title('Volume Distribution')
plt.subplot(1, 3, 2)
sns.histplot(df['Turnover'], bins=50, log_scale=True)
plt.title('Turnover Distribution')
plt.subplot(1, 3, 3)
sns.histplot(df['%Deliverable'], bins=50)
plt.title('%Deliverable Distribution')
plt.tight_layout()
plt.savefig('liquidity_distributions.png')
plt.close()

# Task 8: Save Cleaned Data
df.to_csv('cleaned_stock_data.csv', index=False)

print("Phase 1 completed. Outputs saved: cleaned_stock_data.csv, liquidity_stats.csv, stationarity_volatility.csv, correlation_matrix.csv, liquidity_distributions.png, correlation_heatmap.png, volatility_trend_<n>.png, liquidity_<symbol>.png, and acf_<symbol>_volatility.png for all companies.")