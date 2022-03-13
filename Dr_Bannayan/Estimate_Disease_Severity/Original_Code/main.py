#source_us_no_inoc_susc.csv & no individual field

import datetime
import itertools

import numpy as np
import openpyxl as openpyxl
import pandas as pd

# from tcc_s3 import S3

# from global_vars import S3_BUCKET, PROCESS_MODEL_PATH
from disease_mechanistic_functions_original import blizzard_2_legacy, run_locationId_r_stella

# s3 = S3()

# # data paths
# INPUT_DATA_PATH = S3_BUCKET + PROCESS_MODEL_PATH + "input-data/"
# OUTPUT_DATA_PATH = S3_BUCKET + PROCESS_MODEL_PATH + "output-data/"


# # Download input files from s3
# s3.download(INPUT_DATA_PATH + "Parameters_Corn.xlsx")
# s3.download(INPUT_DATA_PATH + "Parameters_Soy.xlsx")

# Load auxiliary files

# Corn tuning parameters
ip_t_cof_corn = pd.read_excel("Parameters_Corn.xlsx", engine="openpyxl", sheet_name=0, header=None)
p_t_cof_corn = pd.read_excel("Parameters_Corn.xlsx", engine="openpyxl", sheet_name=1, header=None)
rc_t_input_corn = pd.read_excel(
    "Parameters_Corn.xlsx", engine="openpyxl", sheet_name=2, header=None
)
dvs_8_input_corn = pd.read_excel(
    "Parameters_Corn.xlsx", engine="openpyxl", sheet_name=3, header=None
)
rc_a_input_corn = pd.read_excel(
    "Parameters_Corn.xlsx", engine="openpyxl", sheet_name=4, header=None
)
fungicide_corn = pd.read_excel("Parameters_Corn.xlsx", engine="openpyxl", sheet_name=6, header=None)
fungicide_residual_corn = pd.read_excel(
    "Parameters_Corn.xlsx", engine="openpyxl", sheet_name=7, header=None
)

# Soybean tuning parameters
ip_t_cof_soy = pd.read_excel("Parameters_Soy.xlsx", engine="openpyxl", sheet_name=0, header=None)
p_t_cof_soy = pd.read_excel("Parameters_Soy.xlsx", engine="openpyxl", sheet_name=1, header=None)
rc_t_input_soy = pd.read_excel("Parameters_Soy.xlsx", engine="openpyxl", sheet_name=2, header=None)
dvs_8_input_soy = pd.read_excel("Parameters_Soy.xlsx", engine="openpyxl", sheet_name=3, header=None)
rc_a_input_soy = pd.read_excel("Parameters_Soy.xlsx", engine="openpyxl", sheet_name=4, header=None)
fungicide_soy = pd.read_excel("Parameters_Soy.xlsx", engine="openpyxl", sheet_name=6, header=None)
fungicide_residual_soy = pd.read_excel(
    "Parameters_Soy.xlsx", engine="openpyxl", sheet_name=7, header=None
)


# Historical Data
# read pre-calculated historical data (summarized) - for specific fields

historical = pd.read_csv("Data.csv",
#historical = pd.read_csv("Historical_Calculated_Data_Merged_Completed.csv",
# historical = pd.read_csv(
#     #INPUT_DATA_PATH + "Historical_Calculated_Data_Merged_Completed.csv",
#     "s3://com.climate.production.users/team/science/modeling-science/corn/disease-risk-modeling/models/process-models/output/final_corn.csv",
    encoding="latin-1",
    # index_col=0,
)


info = pd.read_csv("Info.csv",
    encoding="latin-1",
    # index_col=0,
)


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

genetic_mechanistic_list = [
    "Susceptible",
    "Moderate",
    "Resistant",
]

# Run specific fields.
fields_to_run = loc_historical_unique["Field"].unique()


# This is just for the parametrized version. Will need to be adjusted to something more reasonable.
fungicide_inputs_full = pd.DataFrame()
fungicide_inputs_full["spray_number"] = [1, 2, 3]
fungicide_inputs_full["spray_moment"] = [30, 45, 60] # min 0
fungicide_inputs_full["spray_eff"] = [0.5, 0.5, 0.5] # min 0, max 1

################### RUN MODEL #############################

for crop_mechanistic, number_applications, genetic_mechanistic in itertools.product(
    crop_mechanistic_list, number_applications_list, genetic_mechanistic_list
):
    # Generate fungicide application table.
    if number_applications > 0:
        # Filter to only the fungicide applications used.
        using_fungicide = True
        fungicide_inputs = fungicide_inputs_full.query("spray_number <= @number_applications").copy()
    else:
        using_fungicide = False
        fungicide_inputs = pd.DataFrame()
        #Mo
        #all_results = pd.DataFrame()

    #################### HISTORICAL DATA MANIPULATION ############
    #model_origin = "2017-12-31" if crop_mechanistic == "Corn" else "2018-01-01"  # TODO: Why?: Original
    model_origin = "2015-12-31" if crop_mechanistic == "Corn" else "2015-12-31" 
    model_origin = pd.to_datetime(model_origin)
    loc_historical = loc_historical_unique[loc_historical_unique["Crop"] == crop_mechanistic]
    historical_input = historical[
        historical["Field"].isin(fields_to_run) & (historical["Crop"] == crop_mechanistic)
    ].copy()
    historical_input["DOY"] = pd.to_timedelta(historical_input["DOY"], unit="d")
    historical_input["time"] = (historical_input["DOY"] + model_origin).dt.strftime("%Y-%m-%d")
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
        [
            "locationId",
            "latitude",
            "longitude",
            "date",
            "DOY",
            "precip",
            "maxtemp",
            "mintemp",
            "avgwindspeed",
            "GDU",
        ]
    ]
    
    historical_input_blizz_raw = historical_input_blizz_prev.copy()
    historical_input_blizz = historical_input_blizz_prev.copy()    
    number_of_repeat_year = 4
    
    for i in range(1, number_of_repeat_year + 1):
        
        historical_input_blizz_prev_modified = historical_input_blizz_raw.copy()
        
        historical_input_blizz_prev_modified["date"] = (
            pd.to_datetime(historical_input_blizz_prev_modified["date"]) + pd.Timedelta(i * 365, "d")
        ).dt.strftime("%Y%m%d")
        
        historical_input_blizz = pd.concat(
            [historical_input_blizz, historical_input_blizz_prev_modified],
            axis=0,
            ignore_index=True,
        )
    
    historical_input_blizz = historical_input_blizz.drop_duplicates(subset=["locationId", "date"])

    if crop_mechanistic == "Corn":
        results = run_locationId_r_stella(
            all_fields_weather=historical_input_blizz,
            info=info,
            ip_t_cof=ip_t_cof_corn,
            p_t_cof=p_t_cof_corn,
            rc_t_input=rc_t_input_corn,
            rc_a_input=rc_a_input_corn,
            dvs_8_input=dvs_8_input_corn,
            p_opt=p_opt,
            inocp=10,
            rrlex_par=rrlex_par,
            rc_opt_par=0.267,
            ip_opt=14,
            is_fungicide=using_fungicide,
            fungicide=fungicide_inputs,
            fungicide_residual=fungicide_residual_corn,
            #MO
#             crop_mechanistic=crop_mechanistic,
#             number_applications=number_applications,
#             genetic_mechanistic=genetic_mechanistic
        )
    else:
        results = run_locationId_r_stella(
            all_fields_weather=historical_input_blizz,
            info=info,
            ip_t_cof=ip_t_cof_soy,
            p_t_cof=p_t_cof_soy,
            rc_t_input=rc_t_input_soy,
            rc_a_input=rc_a_input_soy,
            dvs_8_input=dvs_8_input_soy,
            p_opt=p_opt,
            inocp=10,
            rrlex_par=0.01,
            rc_opt_par=rc_opt_par,
            ip_opt=28,
            is_fungicide=using_fungicide,
            fungicide=fungicide_inputs,
            fungicide_residual=fungicide_residual_soy,
            #Mo
#             crop_mechanistic=crop_mechanistic,
#             number_applications=number_applications,
#             genetic_mechanistic=genetic_mechanistic
        )
    
#     results["CropMechanistic"] = crop_mechanistic
#     results["NumberApplications"] = number_applications
#     results["GeneticMechanistic"] = genetic_mechanistic
#     results["PlantingDate"] = datetime.datetime.strptime(date, "%Y-%m-%d").strftime("%Y%m%d")
   #     all_results = pd.concat([all_results, results])
    
    spray_code = (
        f"-{'-'.join(fungicide_inputs['spray_moment'].astype(str).values)}"
        if using_fungicide
        else ""
    )

    # csv_path = f"./results/{crop_mechanistic}_{number_applications}-app{spray_code}_{genetic_mechanistic}_{date}.csv"
    csv_path = f"./results/{crop_mechanistic}_{number_applications}-app{spray_code}_{genetic_mechanistic}.csv"
    # csv_path = f"{OUTPUT_DATA_PATH}{crop_mechanistic}_{number_applications}-app{spray_code}_{genetic_mechanistic}_{date}.csv"
    
    print(csv_path)

    print (results.head())
    #results.sort_values("locationId", inplace=True)
    # save to local directory
    # results.to_csv(f"{crop_mechanistic}_{number_applications}-app{spray_code}_{genetic_mechanistic}_{date}.csv", index=None)

    if len(results) != 0:
        results.sort_values("locationId").to_csv(csv_path, index=None)
    