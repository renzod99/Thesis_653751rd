import logging
import pandas as pd

from filter.filter_classes.KF.stat import KF
from thesis_data.btc.data.data_bitcoin import import_data
from data.data_utility import data_split, scale_quantity

from filter.hyperparams.hyperparameter2 import mml
from filter.filter_classes.KF.stat import KFInterpreter
from filter.filter_classes.KF.stat.kernel_setupG import process_kernel_string, construct_kernel_ss_mats

logging.basicConfig(
    level=logging.INFO,                    
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"          
)

logger = logging.getLogger(__name__)

data = import_data(frequency="weekly")

train, test = data_split(data, train_percentage=0.9, n_obs=261)
scaler = 100
train, test = scale_quantity((train, test), scaler)
print(len(train), len(test), len(train) + len(test))

kernel_str = "Add(matern32(), exponential(), white_noise())"
descriptor, kernel_metadata = process_kernel_string(kernel_str)
file_path = "thesis_data/btc/data/hyperparameter_config_g.yaml"

hyperparams, result = mml(train, test, kernel_str, bounds_file_path=file_path, number_of_init=5)
for key, value in hyperparams.items():
    print(value)
ss_mats = construct_kernel_ss_mats(descriptor, hyperparams)
kf = KF.KalmanFilter(ss_mats)
kf.run_filter(train)
horizon = 1
kf.run_rolling_forecast(test, horizon)

kfi = KFInterpreter.KFInterpreter(kf, transformation=scaler)
train, test = scale_quantity((train, test), 1/scaler)
errors = kfi.return_forecast_errors_rolling(test, return_full=True)
individual, summary = errors[0]

# print(f"individual errors: \n{individual} \nsummary: \n{summary}")
# print(f"log return summary: \n{data["quantity"].describe()}")
print(f"summary: \n{summary}")

kfi.plot_results_rolling(train, test)