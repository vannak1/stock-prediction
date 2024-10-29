import numpy as np

def feature_engineering(df):
    """
    Create additional features for the model.
    """
    # Price Momentum
    df['return_1w'] = df['close'].pct_change(5)
    df['return_1m'] = df['close'].pct_change(21)

    # Handle infinite values resulting from pct_change
    df['return_1w'] = df['return_1w'].replace([np.inf, -np.inf], np.nan)
    df['return_1m'] = df['return_1m'].replace([np.inf, -np.inf], np.nan)

    # Volume Indicators
    df['vol_avg_1m'] = df['volume'].rolling(window=21).mean()
    df['vol_avg_3m'] = df['volume'].rolling(window=63).mean()

    # Volatility Measures
    df['volatility'] = df['close'].rolling(window=21).std()

    # Moving Averages
    df['ma_20'] = df['close'].rolling(window=20).mean()
    df['ma_50'] = df['close'].rolling(window=50).mean()

    # Lag Features
    df['close_lag_1'] = df['close'].shift(1)
    df['close_lag_2'] = df['close'].shift(2)

    # Replace infinite values with NaN in the entire DataFrame
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Drop rows with NaN values
    df.dropna(inplace=True)

    return df


def label_data(df):
    """
    Label the data where the stock increases by more than 3x within 6 months.
    """
    df['future_max'] = df['close'].rolling(window=126).max().shift(-126)
    df['target'] = np.where((df['future_max'] / df['close']) >= 3, 1, 0)
    
    # Remove rows where future data is not available
    df = df.dropna(subset=['target'])
    df = df.reset_index(drop=True)
    
    return df