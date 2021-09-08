import pandas as pd

from disease_mechanistic_functions import calc_rain_score

df = pd.read_excel(
    'Dr_Bannayan/Disease/data/sample_weather_data.xlsx',
    engine="openpyxl",
    index_col=None
)

df = df[df["ID"] == 40745]

precip_boolean_series = df["precipitation"] >= 2

rain_score = calc_rain_score(
    day=1,
    one_field_precip_bool=precip_boolean_series    
)