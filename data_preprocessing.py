from sklearn.preprocessing import MinMaxScaler

def preprocess_data(df):
    """
    Clean and normalize the data.
    """
    df = df.dropna()
    df = df.reset_index(drop=True)
    
    # Feature Scaling
    scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform(df[['open', 'high', 'low', 'close', 'volume']])
    df[['open', 'high', 'low', 'close', 'volume']] = scaled_features
    
    return df, scaler

