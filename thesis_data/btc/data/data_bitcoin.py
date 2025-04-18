import pandas as pd
import numpy as np

def import_data(frequency='daily'):
    file_path = 'thesis_data/btc/data/btc_usd.csv'
    df = pd.read_csv(file_path)
    df.rename(columns={"price": "price", "snapped_at": "time"}, inplace=True)
    df["time"] = pd.to_datetime(df["time"]).dt.date
    df["time"] = pd.to_datetime(df["time"])
    
    if frequency == 'daily':
        df["time_diff"] = (df["time"].shift(-1) - df["time"]).dt.days
    elif frequency == 'weekly':
        df = df.resample('W', on='time', origin='start').mean().reset_index()
        df["time_diff"] = (df["time"].shift(-1) - df["time"]).dt.days / 7
    elif frequency == 'monthly':
        df = df.resample('ME', on='time', origin="start").mean().reset_index()
        df["time_diff"] = (df["time"].shift(-1) - df["time"]).dt.days // 28
   
    # df["quantity"] =  np.log(df["price"])
    df["quantity"] =  np.log(df["price"] / df["price"].shift(1))
    df.loc[df.index[-1], "time_diff"] = 1
    return df[["quantity", "time", "time_diff", "price"]]