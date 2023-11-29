import numpy as np
import pandas as pd

from typing import Union, Dict, List, Any

from src.utils import *


primitive_or_col = lambda t: Union[t, pd.Series]
int_or_col = primitive_or_col(int)
str_or_col = primitive_or_col(str)


def process_year(
    data_year: int_or_col, base_year=0, year_tres=2001
) -> Dict[str, List[Union[int, str, float]]]:
    old_feature = (data_year < 2001).astype(int)
    new_data_year = data_year - base_year
    new_data_year.loc[old_feature == 1] = 0
    return {"year": new_data_year.values, "old": old_feature.values}


def process_mileage(data_mileage: int_or_col, filling: float):
    data_mileage = data_mileage.copy()
    data_mileage = clear_feature(data_mileage)
    data_mileage = data_mileage.fillna(filling)
    return {"mileage": data_mileage.values * MILES_TO_KM / GALLONS_TO_LITERS}


def process_engine(data_engine: int_or_col, filling: float):
    data_engine = data_engine.copy()
    data_engine = clear_feature(data_engine)
    data_engine = data_engine.fillna(filling)
    keys = ["engine_1000_1250", "engine_1250_1750", "engine_1750_inf"]
    return {k: v for k, v in zip(keys, categorize_engine(data_engine)[1:])}


def process_max_power(data_max_power: int_or_col, filling: float):
    data_max_power = data_max_power.copy()
    data_max_power = clear_feature(data_max_power)
    data_max_power = data_max_power.fillna(filling)
    keys = ["max_power_82_100", "max_power_100_145", "max_power_145_inf"]
    return {k: v for k, v in zip(keys, categorize_max_power(data_max_power)[1:])}


def process_torque(data_torque: int_or_col, filling: List[float]):
    data_torque = data_torque.copy()
    data_split_torque = split_torque(data_torque)
    data_torque = data_split_torque[0].fillna(filling[0])
    data_max_rpm_torque = data_split_torque[1].fillna(filling[1])
    return {
        "log_torque": np.log1p(data_torque.values),
        "max_torque_rpm": data_max_rpm_torque.values,
    }


def process_fuel(data_fuel: str_or_col):
    diesel = "Diesel"
    petrol = "Petrol"
    data_fuel = data_fuel.copy()
    data_fuel[(data_fuel != diesel) & (data_fuel != petrol)] = "other"
    return {"fuel": data_fuel.values}


def process_seats(data_seats: int_or_col, filling: int):
    data_seats = data_seats.copy()
    data_seats = data_seats.fillna(filling)
    return {"seats": data_seats.values}


def identity_process(data, col_name: str):
    return {col_name: data.values}


def process_car_name(data_name: str_or_col):
    data_name = data_name.copy()
    data_name = data_name.str.lower().str.split()
    car_makes = get_car_makes()
    makes = data_name.apply(car_makes)
    return {"car_make": makes.values}


feature_processors = {
    "year": process_year,
    "mileage": process_mileage,
    "engine": process_engine,
    "max_power": process_max_power,
    "torque": process_torque,
    "fuel": process_fuel,
    "seats": process_seats,
    "name": process_car_name,
}

fillable_features = {"mileage", "engine", "max_power", "seats"}
features_with_bonuses = {"year"}


def process_features(df: pd.DataFrame, feature_dict):
    processed_features = {}
    for col in df.columns:
        if col in feature_processors:
            if col in fillable_features:
                processed_features.update(
                    feature_processors[col](df[col], feature_dict["medians"][col])
                )
            elif col == "torque":
                processed_features.update(
                    feature_processors[col](
                        df[col],
                        [
                            feature_dict["medians"]["torque"],
                            feature_dict["medians"]["max_torque_rpm"],
                        ],
                    )
                )
            elif col in features_with_bonuses:
                processed_features.update(
                    feature_processors[col](df[col], **feature_dict[col])
                )
            else:
                processed_features.update(feature_processors[col](df[col]))
        else:
            processed_features.update(identity_process(df[col], col))
    return pd.DataFrame.from_dict(processed_features)


def preprocess_collection(items: List[Any]):
    return pd.DataFrame([item.dict() for item in items])