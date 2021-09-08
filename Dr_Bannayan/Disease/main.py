import datetime
import itertools

import numpy as np
import pandas as pd

from disease_mechanistic_functions import blizzard_2_legacy, run_locationId_r_stella

# Load auxiliary files
 
# Corn tuning parameters
ip_t_cof_corn = pd.read_excel('Parameters_Corn.xlsx', engine="openpyxl", sheet_name=0, header=None)
p_t_cof_corn = pd.read_excel('Parameters_Corn.xlsx', engine="openpyxl", sheet_name=1, header=None)
rc_t_input_corn = pd.read_excel('Parameters_Corn.xlsx', engine="openpyxl", sheet_name=2, header=None)
dvs_8_input_corn = pd.read_excel('Parameters_Corn.xlsx', engine="openpyxl", sheet_name=3, header=None)
rc_a_input_corn = pd.read_excel('Parameters_Corn.xlsx', engine="openpyxl", sheet_name=4, header=None)
fungicide_corn = pd.read_excel('Parameters_Corn.xlsx', engine="openpyxl", sheet_name=6, header=None)
fungicide_residual_corn = pd.read_excel('Parameters_Corn.xlsx', engine="openpyxl", sheet_name=7, header=None)

# Soybean tuning parameters
ip_t_cof_soy = pd.read_excel('Parameters_Soy.xlsx', engine="openpyxl", sheet_name=0, header=None)
p_t_cof_soy = pd.read_excel('Parameters_Soy.xlsx', engine="openpyxl", sheet_name=1, header=None)
rc_t_input_soy = pd.read_excel('Parameters_Soy.xlsx', engine="openpyxl", sheet_name=2, header=None)
dvs_8_input_soy = pd.read_excel('Parameters_Soy.xlsx', engine="openpyxl", sheet_name=3, header=None)
rc_a_input_soy = pd.read_excel('Parameters_Soy.xlsx', engine="openpyxl", sheet_name=4, header=None)
fungicide_soy = pd.read_excel('Parameters_Soy.xlsx', engine="openpyxl", sheet_name=6, header=None)
fungicide_residual_soy = pd.read_excel('Parameters_Soy.xlsx', engine="openpyxl", sheet_name=7, header=None)


# Historical Data
# read pre-calculated historical data (summarized) - for specific fields
historical = pd.read_csv('./csv_files/Historical_Calculated_Data_Merged.csv', encoding="latin-1", index_col=0)
# TD changed name for MD
historical["Area"] = historical["Area"].str.replace("TD", "MD")
loc_historical_unique = historical.drop_duplicates(subset="Field")


################### MOCKED USER INPUTS #############################
# Parametrize
crop_mechanistic_list = [
    "Corn",  # Disease: "Northern Leaf Blight"  
    "Soy",  # Disease: "Asian Soybean Rust"
]
number_applications_list = [
    0,
    1,
    2,
    3,
]  # 0 - 5

# Genetic susceptibility to disease.
genetic_mechanistic_list = [
    "Susceptible",
    "Moderate",
    "Resistant",
]

# Planting date
date_list = ["2018-10-31"]  # Only 2018 for now.

# Run specific fields.
fields_to_run = loc_historical_unique["Field"].unique()


# This is just for the parametrized version. Will need to be adjusted to something more reasonable.
fungicide_inputs_full = pd.DataFrame()
fungicide_inputs_full["spray_number"] = [1, 2, 3]
fungicide_inputs_full["spray_moment"] = [30, 45, 60]  # min 0
fungicide_inputs_full["spray_eff"] = [0.5, 0.5, 0.5]  # min 0, max 1


################### RUN MODEL #############################

for crop_mechanistic, number_applications, genetic_mechanistic, date in itertools.product(crop_mechanistic_list, number_applications_list, genetic_mechanistic_list, date_list):
    print(crop_mechanistic, number_applications, genetic_mechanistic, date)
    
    # Generate fungicide application table.
    if number_applications > 0:
        # Filter to only the fungicide applications used.
        using_fungicide = True
        fungicide_inputs = fungicide_inputs_full[fungicide_inputs_full["spray_number"] <= number_applications]
    else:
        using_fungicide = False
        fungicide_inputs = pd.DataFrame()
    
    
    


    #################### HISTORICAL DATA MANIPULATION ############
    model_origin = "2017-12-31" if crop_mechanistic == "Corn" else "2018-01-01"  # TODO: Why?
    model_origin = pd.to_datetime(model_origin)
    loc_historical = loc_historical_unique[loc_historical_unique["Crop"] == crop_mechanistic]
    historical_input = historical[historical["Field"].isin(fields_to_run) & (historical["Crop"] == crop_mechanistic)].copy()
    historical_input["DOY"] = pd.to_timedelta(historical_input["DOY"], unit="d")
    historical_input["time"] = (historical_input["DOY"] + model_origin).dt.strftime('%Y-%m-%d')
    historical_input["ID"] = historical["Field"]

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

    historical_input_blizz_prev = blizzard_2_legacy(historical_input, crop=crop_mechanistic)
    historical_input_blizz_prev = historical_input_blizz_prev[
        ['locationId', 'latitude', 'longitude', 'date', 'DOY', 'precip', 'maxtemp', 'mintemp', 'avgwindspeed', 'GDU']
    ]
    historical_input_blizz_prev_modified = historical_input_blizz_prev.copy()
    historical_input_blizz_prev_modified["date"] = (
        pd.to_datetime(historical_input_blizz_prev_modified["date"])
        + pd.Timedelta(365, "d")
    ).dt.strftime("%Y%m%d")
    historical_input_blizz = pd.concat([historical_input_blizz_prev, historical_input_blizz_prev_modified], axis=0, ignore_index=True)
    historical_input_blizz = historical_input_blizz.drop_duplicates(subset=["locationId", "date"])

    stella_date = datetime.datetime.strptime(date, "%Y-%m-%d").strftime("%Y%m%d")

    if crop_mechanistic == "Corn":
        results = run_locationId_r_stella(
            all_fields_weather = historical_input_blizz,
            date = stella_date,
            ip_t_cof = ip_t_cof_corn,
            p_t_cof = p_t_cof_corn,
            rc_t_input = rc_t_input_corn,
            rc_a_input = rc_a_input_corn,
            dvs_8_input = dvs_8_input_corn,
            p_opt = p_opt,
            inocp = 10,
            rrlex_par = rrlex_par,
            rc_opt_par = 0.267,
            ip_opt = 14,
            is_fungicide = using_fungicide,
            fungicide = fungicide_inputs,
            fungicide_residual = fungicide_residual_corn,
        )
    else:
        results = run_locationId_r_stella(
            all_fields_weather = historical_input_blizz,
            date = stella_date,
            ip_t_cof = ip_t_cof_soy,
            p_t_cof = p_t_cof_soy,
            rc_t_input = rc_t_input_soy,
            rc_a_input = rc_a_input_soy,
            dvs_8_input = dvs_8_input_soy,
            p_opt = p_opt,
            inocp = 10,
            rrlex_par = 0.01,
            rc_opt_par = rc_opt_par, https://conference.ifas.ufl.edu/pest/index.html
            ip_opt = 28,
            isFungicide = using_fungicide,
            fungicide = fungicide_inputs,
            fungicide_residual = fungicide_residual_soy
        )
    spray_code = f"-{'-'.join(fungicide_inputs['spray_moment'].astype(str).values)}" if using_fungicide else ""
    csv_path = f"/mnt/py_results/{crop_mechanistic}_{number_applications}-app{spray_code}_{genetic_mechanistic}_{date}.csv"

    print(csv_path)
    print(results.head())
    results.sort_values("locationId").to_csv(csv_path, index=None)    


