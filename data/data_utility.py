import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def data_split(df:pd.DataFrame, train_percentage:float, n_obs:int=None, val_percentage:float=None):
    if n_obs:
        df = df.tail(n_obs)
    split_index_train = int(len(df) * train_percentage)
    train = df.iloc[:split_index_train]
    test = df.iloc[split_index_train:]

    if val_percentage:
        split_index_val = int(len(train) * (1-val_percentage))
        val = train.iloc[split_index_val:]
        train = train.iloc[:split_index_val]
        return train, val, test 
    return train, test

def scale_quantity(data, factor: float):
    def scale(df):
        df = df.copy()
        df['quantity'] = df['quantity'] * factor
        return df

    if isinstance(data, tuple):
        return tuple(scale(df) for df in data)
    else:
        return scale(data)