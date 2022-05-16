# -----------------------------------------------------------------------------
# Calculate Disease Severity
# -----------------------------------------------------------------------------

from typing import Dict, Optional, Tuple, Union, List
import math

import numpy as np
import pandas as pd
import datetime
import itertools

from .location_data import *
from .corp_parameters import *
from .eds import *

def calculate_disease_severity(
    
    data_path: str,
    info_path: str,
    number_of_repeat_year: int = 1,
       
    number_applications_list: List[int] = [0, 1, 2, 3],
    genetic_mechanistic_list: List[str] = ["Susceptible", "Moderate", "Resistant"],
    
    required_para = None,
    
    user_crop_parameters: bool = False,    
    crop_mechanistic: List[str] = None,   
    crop_parameters_path: List[str] = None,
    
    output_columns: List[str] = ["Sev50%", "SevMAX", "AUC"],
    
    path_result = None
):
    
    all_results = []

    fungicide_inputs_full = required_para["fungicide_inputs_full"]

    data = location_data(
        data_path = data_path,
        info_path = info_path,
        number_of_repeat_year = number_of_repeat_year
    )
    
    if user_crop_parameters:
        crop_para = corp_parameters(
            crop_mechanistic = crop_mechanistic,
            crop_parameters_path = crop_parameters_path,
        )
    else:
        crop_para = CROP_PARAMETERS

    for id in data["info_id"].unique():
        df = data[data["info_id"] == id]
        planting_date_list = [pd.to_datetime(dt).strftime("%Y%m%d") for dt in df["planting_date"].unique()]
        
        for number_applications, genetic_mechanistic, date in itertools.product(number_applications_list, genetic_mechanistic_list, planting_date_list):

            if number_applications > 0:
                using_fungicide = True
                fungicide_inputs = fungicide_inputs_full[fungicide_inputs_full["spray_number"] <= number_applications]
            else:
                using_fungicide = False
                fungicide_inputs = pd.DataFrame()
            
            crop_para_selected = crop_para[df["Crop"].unique()[0]]
            
            df = df[df["date"] >= date].copy()
            
            if len(df) == 0:
                print(f"Location {id} NOT USED. No dates in range.")
                continue
            
            df["Day"] = (
                df["DOY"] - df["DOY"].iloc[0]
            ).dt.days + 1
            
            
            field_results, n_day = eds(
                one_field_weather=df,
                ip_t_cof=crop_para_selected["ip_t_cof"],
                p_t_cof=crop_para_selected["p_t_cof"],
                rc_t_input=crop_para_selected["rc_t_input"],
                dvs_8_input=crop_para_selected["dvs_8_input"],
                rc_a_input=crop_para_selected["rc_a_input"],
                p_opt=required_para["genetic_mechanistic_para"][genetic_mechanistic]["p_opt"],
                inocp=10,
                rrlex_par=required_para["genetic_mechanistic_para"][genetic_mechanistic]["rrlex_par"],
                rc_opt_par=required_para["genetic_mechanistic_para"][genetic_mechanistic]["rc_opt_par"],
                ip_opt=14 if df["Crop"].unique()[0] == "Corn" else 28,
                is_fungicide=using_fungicide,
                fungicide=fungicide_inputs,
                fungicide_residual=crop_para_selected["fungicide_residual"],
                days_after_planting=df["obs_planting_delta"].unique()[0]
            )
            
            # Output information
            start_date = df["date"].iloc[0]
            end_date = df["date"].iloc[n_day - 1]
            result_location = {}
            result_location["locationId"] = id
            result_location["Date1"] = start_date
            result_location["Date2"] = end_date
            result_location["N_Days"] = (
                pd.to_datetime(end_date) - pd.to_datetime(start_date)
            ).days
            result_location["latitude"] = df["latitude"].iloc[0]
            result_location["longitude"] = df["longitude"].iloc[0]
            result_location["Sev50%"] = field_results["Sev"].median()
            result_location["SevMAX"] = field_results["Sev"].max()
            nonzero_sev = field_results[field_results["Sev"] != 0]["Sev"]
            
            if len(nonzero_sev):
                result_location["AUC"] = np.trapz(nonzero_sev)
            else:
                result_location["AUC"] = 0
            
            result_location["number_applications"] = number_applications
            result_location["genetic_mechanistic"] = genetic_mechanistic
            result_location["crop"] = df["Crop"].unique()[0]
            
            all_results.append(result_location)

    all_results = pd.DataFrame.from_dict(all_results)
    
    all_results.sort_values(
        by=["locationId", "crop", "Date1", "number_applications", "genetic_mechanistic"],
        inplace=True
    )    
    
    all_results = all_results[["locationId", "Date1", "Date2", "N_Days", "latitude", "longitude", "number_applications", "genetic_mechanistic", "crop"] + output_columns]   
    
    return all_results