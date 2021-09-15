import pandas as pd
import datetime


from disease_mechanistic_functions import blizzard_2_legacy, r_stella


# -----------------------------------------------------------------------------
Parameters_Corn_Path = "Dr_Bannayan/Disease/data/Parameters_Corn.xlsx"

ip_t_cof_corn = pd.read_excel(
    Parameters_Corn_Path,
    engine="openpyxl",
    sheet_name=0,
    header=None
)

p_t_cof_corn = pd.read_excel(
    Parameters_Corn_Path,
    engine="openpyxl",
    sheet_name=1,
    header=None
)

rc_t_input_corn = pd.read_excel(
    Parameters_Corn_Path,
    engine="openpyxl",
    sheet_name=2,
    header=None
)

dvs_8_input_corn = pd.read_excel(
    Parameters_Corn_Path,
    engine="openpyxl",
    sheet_name=3,
    header=None
)

rc_a_input_corn = pd.read_excel(
    Parameters_Corn_Path,
    engine="openpyxl",
    sheet_name=4,
    header=None
)

fungicide_residual_corn = pd.read_excel(
    Parameters_Corn_Path,
    engine="openpyxl",
    sheet_name=7,
    header=None
)
# -----------------------------------------------------------------------------



# -----------------------------------------------------------------------------
date_list = ["2018-10-31"]

date = datetime.datetime.strptime(date_list[0], "%Y-%m-%d").strftime("%Y%m%d")
# -----------------------------------------------------------------------------



# -----------------------------------------------------------------------------
historical = pd.read_csv(
    "Dr_Bannayan/Disease/data/Historical_Calculated_Data_Merged.csv",
    encoding="utf-8", 
    index_col=None
)

crop_mechanistic = "Corn"

fields_to_run = [40745]

model_origin = "2017-12-31" if crop_mechanistic == "Corn" else "2018-01-01"
model_origin = pd.to_datetime(model_origin)

historical_input = historical[(historical["Field"].isin(fields_to_run)) & (historical["Crop"] == crop_mechanistic)].copy()
historical_input["DOY"] = pd.to_timedelta(historical_input["DOY"], unit="d")
historical_input["time"] = (historical_input["DOY"] + model_origin).dt.strftime('%Y-%m-%d')
historical_input["ID"] = historical["Field"]

historical_input_blizz_prev = blizzard_2_legacy(
    df=historical_input,
    crop=crop_mechanistic
)

historical_input_blizz_prev = historical_input_blizz_prev[
    ['locationId', 'latitude', 'longitude', 'date', 'DOY', 'precip', 'maxtemp', 'mintemp', 'avgwindspeed', 'GDU']
]

historical_input_blizz_prev_modified = historical_input_blizz_prev.copy()

historical_input_blizz_prev_modified["date"] = (
    pd.to_datetime(historical_input_blizz_prev_modified["date"])
    + pd.Timedelta(365, "d")
).dt.strftime("%Y%m%d")

historical_input_blizz = pd.concat(
    [historical_input_blizz_prev, historical_input_blizz_prev_modified],
    axis=0,
    ignore_index=True
)

historical_input_blizz = historical_input_blizz.drop_duplicates(subset=["locationId", "date"])

historical_input_blizz["Temperature"] = historical_input_blizz[["maxtemp", "mintemp"]].mean(
    axis=1
)

historical_input_blizz["Rain"] = (
    historical_input_blizz["precip"] >= 2
)

for i, (locale, one_field_weather) in enumerate(historical_input_blizz.groupby("locationId")):
    print(f"Running Location {i} Id = {locale}")
    
    one_field_weather = one_field_weather[one_field_weather["date"] >= date].copy()
    
    if len(one_field_weather) == 0:
        print(f"Location {i} Id = {locale}  NOT USED. No Dates in Range.")
        continue
    
    one_field_weather["Day"] = (
        one_field_weather["DOY"] - one_field_weather["DOY"].iloc[0]
    ).dt.days + 1
# -----------------------------------------------------------------------------



# -----------------------------------------------------------------------------
genetic_mechanistic_list = [
    "Susceptible",
    "Moderate",
    "Resistant",
]

genetic_mechanistic = genetic_mechanistic_list[1]

if genetic_mechanistic == "Susceptible":
    p_opt = 7
    rc_opt_par = 0.35
    rrlex_par = 0.1
elif genetic_mechanistic == "Moderate":
    p_opt = 10
    rc_opt_par = 0.25
    rrlex_par = 0.01
else:
    p_opt = 14
    rc_opt_par = 0.15
    rrlex_par = 0.0001
# -----------------------------------------------------------------------------



# -----------------------------------------------------------------------------
fungicide_inputs_full = pd.DataFrame()
fungicide_inputs_full["spray_number"] = [1, 2, 3]
fungicide_inputs_full["spray_moment"] = [30, 45, 60]
fungicide_inputs_full["spray_eff"] = [0.5, 0.5, 0.5]

number_applications_list = [
    0,
    1,
    2,
    3,
]

number_applications = number_applications_list[1]

if number_applications > 0:
    using_fungicide = True
    fungicide_inputs = fungicide_inputs_full[fungicide_inputs_full["spray_number"] <= number_applications]
else:
    using_fungicide = False
    fungicide_inputs = pd.DataFrame()
# -----------------------------------------------------------------------------



# -----------------------------------------------------------------------------
field_results, n_day = r_stella(
    one_field_weather=one_field_weather,
    ip_t_cof=ip_t_cof_corn,
    p_t_cof=p_t_cof_corn,
    rc_t_input=rc_t_input_corn,
    dvs_8_input=dvs_8_input_corn,
    rc_a_input=rc_a_input_corn,
    inocp=10,
    p_opt=p_opt,
    rc_opt_par=0.267,
    rrlex_par=rrlex_par,
    ip_opt=14,
    is_fungicide=using_fungicide,
    fungicide=fungicide_inputs,
    fungicide_residual=fungicide_residual_corn,
)
# -----------------------------------------------------------------------------

print(1)
