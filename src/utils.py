import numpy as np
import pandas as pd

from typing import Union

MILES_TO_KM = 1.60934
GALLONS_TO_LITERS = 3.78541

def categorize_engine(data_engine):
    engine_0_1000 = (data_engine <= 1000).astype(int)
    engine_1000_1250 = ((1000 < data_engine) & (data_engine <= 1250)).astype(int)
    engine_1250_1750 = ((1250 < data_engine) & (data_engine <= 1750)).astype(int)
    engine_1750_inf = (1750 < data_engine).astype(int)
    return [engine_0_1000, engine_1000_1250, engine_1250_1750, engine_1750_inf]


def categorize_max_power(data_max_power):
  max_power_0_82 = (data_max_power <= 82).astype(int)
  max_power_82_100 = ((82 < data_max_power) & (data_max_power <= 100)).astype(int)
  max_power_100_145 = ((100 < data_max_power) & (data_max_power <= 145)).astype(int)
  max_power_145_inf = (145 < data_max_power).astype(int)
  return [max_power_0_82, max_power_82_100, max_power_100_145, max_power_145_inf]