import pandas as pd
import numpy as np
from scipy.stats import zscore


def import_temp_data():
    legend = {
        "YYYYMMDD": "Datum (YYYY=jaar MM=maand DD=dag)",
        "DDVEC": "Vectorgemiddelde windrichting in graden",
        "FHVEC": "Vectorgemiddelde windsnelheid (in 0.1 m/s)",
        "FG": "Etmaalgemiddelde windsnelheid (in 0.1 m/s)",
        "FHX": "Hoogste uurgemiddelde windsnelheid (in 0.1 m/s)",
        "FHXH": "Uurvak waarin FHX is gemeten",
        "FHN": "Laagste uurgemiddelde windsnelheid (in 0.1 m/s)",
        "FHNH": "Uurvak waarin FHN is gemeten",
        "FXX": "Hoogste windstoot (in 0.1 m/s)",
        "FXXH": "Uurvak waarin FXX is gemeten",
        "TG": "Etmaalgemiddelde temperatuur (in 0.1 graden Celsius)",
        "TN": "Minimum temperatuur (in 0.1 graden Celsius)",
        "TNH": "Uurvak waarin TN is gemeten",
        "TX": "Maximum temperatuur (in 0.1 graden Celsius)",
        "TXH": "Uurvak waarin TX is gemeten",
        "T10N": "Minimum temperatuur op 10 cm hoogte (in 0.1 graden Celsius)",
        "T10NH": "6-uurs tijdvak waarin T10N is gemeten",
        "SQ": "Zonneschijnduur (in 0.1 uur)",
        "SP": "Percentage van de langst mogelijke zonneschijnduur",
        "Q": "Globale straling (in J/cm2)",
        "DR": "Duur van de neerslag (in 0.1 uur)",
        "RH": "Etmaalsom van de neerslag (in 0.1 mm)",
        "RHX": "Hoogste uursom van de neerslag (in 0.1 mm)",
        "RHXH": "Uurvak waarin RHX is gemeten",
        "PG": "Etmaalgemiddelde luchtdruk herleid tot zeeniveau (in 0.1 hPa)",
        "PX": "Hoogste uurwaarde van de luchtdruk herleid tot zeeniveau (in 0.1 hPa)",
        "PXH": "Uurvak waarin PX is gemeten",
        "PN": "Laagste uurwaarde van de luchtdruk herleid tot zeeniveau (in 0.1 hPa)",
        "PNH": "Uurvak waarin PN is gemeten",
        "VVN": "Minimum opgetreden zicht",
        "VVNH": "Uurvak waarin VVN is gemeten",
        "VVX": "Maximum opgetreden zicht",
        "VVXH": "Uurvak waarin VVX is gemeten",
        "NG": "Etmaalgemiddelde bewolking (bedekkingsgraad van de bovenlucht in achtsten)",
        "UG": "Etmaalgemiddelde relatieve vochtigheid (in procenten)",
        "UX": "Maximale relatieve vochtigheid (in procenten)",
        "UXH": "Uurvak waarin UX is gemeten",
        "UN": "Minimale relatieve vochtigheid (in procenten)",
        "UNH": "Uurvak waarin UN is gemeten",
        "EV24": "Referentiegewasverdamping (Makkink) (in 0.1 mm)"
    }

    column_names = [
        "STN", "YYYYMMDD", "DDVEC", "FHVEC", "FG", "FHX", "FHXH", "FHN", "FHNH",
        "FXX", "FXXH", "TG", "TN", "TNH", "TX", "TXH", "T10N", "T10NH", "SQ", "SP",
        "Q", "DR", "RH", "RHX", "RHXH", "PG", "PX", "PXH", "PN", "PNH", "VVN", "VVNH",
        "VVX", "VVXH", "NG", "UG", "UX", "UXH", "UN", "UNH", "EV24"
    ]

    file_path = "thesis_data/knmi/data/temperature_data.txt"
    data = pd.read_csv(
        file_path,
        skiprows=52,  # Skip the header and metadata
        delimiter=",",  # Use comma as delimiter
        names=column_names,
        skipinitialspace=True,  # Remove extra spaces
        na_values=[" ", ""],  # Treat empty fields as NaN
    )
    data = data.dropna(how="all")
    return data

def data_prep_daily():
    data = import_temp_data()
    data["YYYYMMDD"] = pd.to_datetime(data["YYYYMMDD"], format="%Y%m%d")
    data_subset = data[["YYYYMMDD", "TG"]]
    data_subset = data_subset.rename(columns={"TG": "quantity"})
    data_subset["time_diff"] = (data_subset["YYYYMMDD"].shift(-1) - data_subset["YYYYMMDD"]).dt.days
    data_subset.set_index('YYYYMMDD', inplace=True)
    return data_subset


def data_prep_weekly():
    data = import_temp_data()
    data["YYYYMMDD"] = pd.to_datetime(data["YYYYMMDD"], format="%Y%m%d")
    data_subset = data[["YYYYMMDD", "TG"]].rename(columns={"TG": "quantity"})
    weekly_data = data_subset.resample('W-SUN', on="YYYYMMDD", origin="start").mean().reset_index()
    weekly_data["YYYYMMDD"] = weekly_data["YYYYMMDD"] - pd.DateOffset(days=6)
    weekly_data["time_diff"] = (weekly_data["YYYYMMDD"].shift(-1) - weekly_data["YYYYMMDD"]).dt.days / 7
    data_subset.set_index('YYYYMMDD', inplace=True)
    return weekly_data

def data_prep_monthly():
    data = import_temp_data()
    data["YYYYMMDD"] = pd.to_datetime(data["YYYYMMDD"], format="%Y%m%d")
    data_subset = data[["YYYYMMDD", "TG"]].rename(columns={"TG": "quantity"})
    monthly_data = data_subset.resample('MS', on="YYYYMMDD").mean().reset_index()
    monthly_data["time_diff"] = (monthly_data["YYYYMMDD"].shift(-1) - monthly_data["YYYYMMDD"]).dt.days // 28
    monthly_data.loc[monthly_data.index[-1], "time_diff"] = 1
    data_subset.set_index('YYYYMMDD', inplace=True)
    return monthly_data