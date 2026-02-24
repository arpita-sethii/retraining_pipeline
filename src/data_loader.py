import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def load_data(filepath: str):
    df = pd.read_csv(filepath)
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    return df

def create_sequences(series: np.ndarray, seq_len: int = 24):
    X, y = [], []
    for i in range(len(series) - seq_len):
        X.append(series[i:i + seq_len])
        y.append(series[i + seq_len])
    return np.array(X), np.array(y)

def get_chunks(filepath: str, seq_len: int = 24):
    df = load_data(filepath)

    # Scale entire series first so all chunks are on same scale
    values = df['OT'].values.reshape(-1, 1)
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(values).flatten()

    # Split into 3 chronological chunks
    # Chunk 1: first 40% → train baseline
    # Chunk 2: next 30% → first evaluation (mild drift)
    # Chunk 3: last 30% → second evaluation (seasonal drift)
    n = len(scaled)
    c1_end = int(n * 0.4)
    c2_end = int(n * 0.7)

    chunk1 = scaled[:c1_end]
    chunk2 = scaled[c1_end:c2_end]
    chunk3 = scaled[c2_end:]

    print(f"Chunk 1 (train baseline) : {len(chunk1)} points")
    print(f"Chunk 2 (eval - mild)    : {len(chunk2)} points")
    print(f"Chunk 3 (eval - drift)   : {len(chunk3)} points")

    # Create sequences for each chunk
    X1, y1 = create_sequences(chunk1, seq_len)
    X2, y2 = create_sequences(chunk2, seq_len)
    X3, y3 = create_sequences(chunk3, seq_len)

    # Reshape for LSTM: (samples, seq_len, features)
    X1 = X1.reshape(-1, seq_len, 1)
    X2 = X2.reshape(-1, seq_len, 1)
    X3 = X3.reshape(-1, seq_len, 1)

    return (X1, y1), (X2, y2), (X3, y3), scaler

def get_train_test(filepath: str, seq_len: int = 24, split: float = 0.8):
    df = load_data(filepath)
    values = df['OT'].values.reshape(-1, 1)
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(values).flatten()

    split_idx = int(len(scaled) * split)
    train_data = scaled[:split_idx]
    test_data = scaled[split_idx:]

    X_train, y_train = create_sequences(train_data, seq_len)
    X_test, y_test = create_sequences(test_data, seq_len)

    X_train = X_train.reshape(-1, seq_len, 1)
    X_test = X_test.reshape(-1, seq_len, 1)

    return X_train, y_train, X_test, y_test, scaler