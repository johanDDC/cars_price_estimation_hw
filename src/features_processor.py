import numpy as np
import pandas as pd

from typing import Union

from utils import *

primitive_or_col = lambda t: Union[t, pd.Series]
int_or_col = primitive_or_col(int)
str_or_col = primitive_or_col(str)


def process_year(data_year: int_or_col, base_year = 0, year_tres = 2001):
    old_feature = (data_year < 2001).astype(int)
    new_data_year = data_year - base_year
    if type(data_year) is int and old_feature:
        new_data_year = 0
    elif type(data_year) is not int:
        new_data_year.loc[old_feature == 1] = 0
    return new_data_year, old_feature


def process_mileage(data_mileage: int_or_col):
    return data_mileage * MILES_TO_KM / GALLONS_TO_LITERS


def process_engine(data_engine: int_or_col):
    return categorize_engine(data_engine)[1:]


def process_max_power(data_max_power: int_or_col):
    return categorize_max_power(data_max_power)[1:]


def process_torque(data_torque: int_or_col):
    return np.log1p(data_torque)


def process_fuel(data_fuel: str_or_col):
    new_data = data_fuel.copy()
    diesel = "Diesel"
    pretrol = "Petrol"
    if type(data_fuel) is str:
        if data_fuel != diesel and data_fuel != pretrol:
            return "other"
        return data_fuel
    new_data[(data_fuel != diesel) & (data_fuel != pretrol)] = "other"
    return new_data