import pandas as pd

from disease_mechanistic_functions import *

df = pd.read_excel(
    "Dr_Bannayan/Disease/data/sample_weather_data.xlsx",
    engine="openpyxl",
    index_col=0
)

df = blizzard_2_legacy(
    df=df,
    crop="Corn"
)