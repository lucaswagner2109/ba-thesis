import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler

import pandas as pd
import numpy as np

def load_data(path:str)->pd.DataFrame:
    # from csv
    df = pd.read_csv(path, header=None)
    # name columns
    cols = ["date", "open", "high", "low", "price", "open_interest", "volume"]
    df = df.rename({i: c for i, c in enumerate(cols)}, axis=1)
    # reorder
    order = ["date", "volume", "open_interest", "open", "high", "low", "price"]
    df = df[order]
    # format date
    df.date = pd.to_datetime(df.date, format="%m/%d/%Y")

    return df

def preprocess_data(data:pd.DataFrame, span:int=63)->pd.DataFrame:    
    # calculate log returns
    data["return"] = np.log(data["price"].iloc[1:] / data["price"].shift(1).iloc[1:]) # log(p_{t} / p_{t-1})
    
    # calculate mean return and volatility
    data["mean_return"] = data["return"].rolling(span=span).mean()
    data["volatility"] = data["return"].rolling(span=span).std()

    data = data.reset_index(drop=True)
    return data

def make_seq_dataset(df, features, target, seq_len, start, end):
    df = df.copy()
    df = df[(df["date"] >= start) & (df["date"] < end)]

    # dtype: list[str]
    if isinstance(features, str):
        features = [features]
    if isinstance(target, str):
        target = [target]
        
    # scaling
    feature_scaler = MinMaxScaler()
    target_scaler = MinMaxScaler()
    
    X_df = feature_scaler.fit_transform(df[features])
    y_df = target_scaler.fit_transform(df[target])
    
    # sequence
    Xs, ys = [], []
    for i in range(seq_len, len(X_df)):
        Xs.append(X_df[i - seq_len:i])
        ys.append(y_df[i])

    # convert to tensor
    X = torch.tensor(np.array(Xs), dtype=torch.float32)
    y = torch.tensor(np.array(ys), dtype=torch.float32)
    
    # check shape
    assert X.shape[1:] == torch.Size([seq_len, len(features)])
    assert y.shape[1:] == torch.Size([len(target)])
    
    # torch dataset
    dataset = TensorDataset(X, y)
    return dataset