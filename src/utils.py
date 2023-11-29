import re
import pickle
import numpy as np
import pandas as pd

from typing import Union

MILES_TO_KM = 1.60934
GALLONS_TO_LITERS = 3.78541
NM_CONST = 9.80665

digits_pattern = re.compile(r"[^\d+.\d*]")


def categorize_engine(data_engine):
    engine_0_1000 = (data_engine <= 1000).astype(int)
    engine_1000_1250 = ((1000 < data_engine) & (data_engine <= 1250)).astype(int)
    engine_1250_1750 = ((1250 < data_engine) & (data_engine <= 1750)).astype(int)
    engine_1750_inf = (1750 < data_engine).astype(int)
    return [
        engine_0_1000.values,
        engine_1000_1250.values,
        engine_1250_1750.values,
        engine_1750_inf.values,
    ]


def categorize_max_power(data_max_power):
    max_power_0_82 = (data_max_power <= 82).astype(int)
    max_power_82_100 = ((82 < data_max_power) & (data_max_power <= 100)).astype(int)
    max_power_100_145 = ((100 < data_max_power) & (data_max_power <= 145)).astype(int)
    max_power_145_inf = (145 < data_max_power).astype(int)
    return [
        max_power_0_82.values,
        max_power_82_100.values,
        max_power_100_145.values,
        max_power_145_inf.values,
    ]


def clear_feature(data) -> pd.Series:
    data = data.str.replace(digits_pattern, "", regex=True)
    lengths = data.str.len()
    critical_idx = lengths[lengths == 0].index
    data.loc[critical_idx] = np.nan
    return data.astype(np.float64)


def split_torque(torque: pd.Series) -> pd.DataFrame:
    data = torque.copy()
    data = data.str.replace(r"[^0-9-. ]", "", regex=True)
    data = data.str.extract(r"(\d+[.|,]?\d*)\s+(\d+-?\d+)")
    data[1] = data[1].str.replace(r"\d+-", "", regex=True)
    data = data.astype(np.float64)
    data["measure"] = torque.str.lower().str.extract(r"(kgm|nm)")
    data.loc[data["measure"] == "kgm", 0] *= NM_CONST
    return data


def get_car_makes():
    makes = None
    with open("assets/car_makes.pickle", "rb") as f:
        makes = pickle.load(f)

    def func(name_list):
        for name in name_list:
            if name in makes:
                return name
            return "no mark"

    return func
