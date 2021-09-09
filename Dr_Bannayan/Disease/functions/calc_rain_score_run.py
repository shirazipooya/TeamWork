import pandas as pd

from disease_mechanistic_functions import calc_rain_score

# Read Pre-calculated Historical Data (Summarized)
historical = pd.read_csv(
    "Dr_Bannayan/Disease/data/Historical_Calculated_Data_Merged.csv",
    encoding="utf-8", 
    index_col=None
)

# Run specific fields.
loc_historical_unique = historical.drop_duplicates(subset="Field")

# Parametrize
crop_mechanistic = "Corn"

# Historical Data Manipulation
model_origin = "2017-12-31" if crop_mechanistic == "Corn" else "2018-01-01"
model_origin = pd.to_datetime(model_origin)

historical_input = historical[(historical["Field"] == 40745) & (historical["Crop"] == crop_mechanistic)].copy()
historical_input["DOY"] = pd.to_timedelta(historical_input["DOY"], unit="d")
historical_input["time"] = (historical_input["DOY"] + model_origin).dt.strftime('%Y-%m-%d')
historical_input["ID"] = historical["Field"]

precip_boolean_series = historical_input["precipitation"] >= 2

rain_score = calc_rain_score(
    day=1,
    one_field_precip_bool=precip_boolean_series    
)

print(rain_score)