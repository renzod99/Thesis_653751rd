import logging
import numpy as np
import pandas as pd

from data.data_utility import convert_OHCL_json_to_df, data_split, scale_quantity
from trading_indicators import indicators
from filter.hyperparams.hyperparameter2 import mml_t
from filter.filter_classes.TFilter import TFilter, TFilterInterpreter
from filter.filter_classes.TFilter.kernelsetupT import process_kernel_string, construct_kernel_ss_mats


logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,  # Set minimum log level to INFO
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(),  # Logs to console
        logging.FileHandler("my_app.log")  # Logs to file
    ]
)

file_path = "data/data/timeseries_data.json"
df = convert_OHCL_json_to_df(file_path=file_path)
df = indicators.generate_columns(df)
df = df.rename(columns={'log_return_close': 'quantity'})

df_subset = df[['quantity', 'time_diff']]
train, test = data_split(df_subset, train_percentage=0.8)
train, test = scale_quantity((train, test), 1000)

print(f"head of df: \n{df.head(5)}\nn_obs: {len(df)}")
print(df_subset["quantity"].describe())
print("Training Set Size:", len(train))
print("Test Set Size:", len(test))

kernel_str = "Add(ratquad(), white_noise())"
bounds_path = "gpss/hyperparameter_config.yaml"
hyperparams, result = mml_t(train, test, kernel_str, bounds_file_path=bounds_path, number_of_init=3)
for key, value in hyperparams.items():
    print(value)

descriptor, kernel_metadata = process_kernel_string(kernel_str)
ss_mats = construct_kernel_ss_mats(descriptor, hyperparams)
kf = TFilter.StudentTFilter(ss_mats)
kf.run_filter(train)
horizon = 1
kf.run_rolling_forecast(test, horizon)
kfi = TFilterInterpreter.TFInterpreter(kf, transformation=(True,1/1000))

train, test = scale_quantity((train, test), 1/1000)
print("mean log return:", test["quantity"].mean())
errors = kfi.return_forecast_errors_rolling(test)
for i, error in enumerate(errors):
    print(f"window: {i+1}, forecast error metrics:\nMAE: {error['mae']}, \nda: {error['da']}")
kfi.plot_results_rolling(train, test, horizon)