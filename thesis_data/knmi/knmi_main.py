from filter.filter_classes.KF.stat import KF
from filter.filter_classes.KF.stat import KFInterpreter
from filter.hyperparams.hyperparameter2 import mml
from thesis_data.knmi.data.data_knmi import data_prep_monthly
from data.data_utility import data_split
from filter.filter_classes.KF.stat.kernel_setupG import process_kernel_string, construct_kernel_ss_mats


data = data_prep_monthly()
train, test = data_split(data, train_percentage=0.8)
print(len(train), len(test), len(train) + len(test))
kernel_str = "Add(constant(), linear(), periodic(), periodic(), white_noise())" 
descriptor, kernel_metadata = process_kernel_string(kernel_str)

file_path = "thesis_data/knmi/data/hyperparameter_config.yaml"
hyperparams, result = mml(train, test, kernel_str, bounds_file_path=file_path,number_of_init=1)
for key, value in hyperparams.items():
    print(value)

ss_mats = construct_kernel_ss_mats(descriptor, hyperparams)
kf = KF.KalmanFilter(ss_mats)
kf.run_filter(train)
kf.run_forecast(test)

kfi = KFInterpreter.KFInterpreter(kf)
errors = kfi.return_forecast_errors(test, return_full=True)
individual, summary = errors

# print(f"individual errors: \n{individual} \nsummary: \n{summary}")
print(f"summary: \n{summary}")

kfi.plot_results(train, test)
