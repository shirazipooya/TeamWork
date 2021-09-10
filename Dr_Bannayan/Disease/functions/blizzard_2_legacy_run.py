import pandas as pd

from disease_mechanistic_functions import blizzard_2_legacy

# read pre-calculated historical data (summarized) - for specific fields
historical = pd.read_csv(
    "Dr_Bannayan/Disease/data/Historical_Calculated_Data_Merged.csv",
    encoding="utf-8", 
    index_col=None
)

# Parametrize
crop_mechanistic = "Corn"

# Historical Data Manipulation
model_origin = "2017-12-31" if crop_mechanistic == "Corn" else "2018-01-01"
model_origin = pd.to_datetime(model_origin)

historical_input = historical[(historical["Field"] == 40745) & (historical["Crop"] == crop_mechanistic)].copy()
historical_input["DOY"] = pd.to_timedelta(historical_input["DOY"], unit="d")
historical_input["time"] = (historical_input["DOY"] + model_origin).dt.strftime('%Y-%m-%d')
historical_input["ID"] = historical["Field"]

# Preprocess Blizzard Weather Data and Calculates Growing Degree Units
df = blizzard_2_legacy(
    df=historical_input,
    crop=crop_mechanistic
)

print(df)