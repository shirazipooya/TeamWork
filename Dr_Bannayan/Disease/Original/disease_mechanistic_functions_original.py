"""
 A set of functions that will be used to estimate disease severity for corn and soybeans

 R Code by Vinicius S Junqueira and Braz Hora
 12-2016

 Translated to Python by Peter Carlson
 05-2021
 Note: This is a minimal translation. It has not been optimized for Python.
"""

from typing import Dict, Optional, Tuple, Union
import math

import numpy as np
import pandas as pd


def blizzard_2_legacy(df: pd.DataFrame, crop: str) -> pd.DataFrame:
    """
    Preprocess Blizzard weather data and calculates Growing Degree Units

    Parameters
    ----------
    df : pd.DataFrame
        Blizzard Weather data
    crop : str
        `Corn` or `Soy`

    Returns
    -------
    df : pd.DataFrame
        Blizzard Weather data with renamed columns, datetime `date`, and GDUs calculated.
    """
    df = df.rename(
        columns={
            "ID": "locationId",
            "time": "date",
            "precipitation": "precip",
            "maximum_temperature": "maxtemp",
            "minimum_temperature": "mintemp",
            "wind_speed": "avgwindspeed",
        }
    )
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y%m%d")
    if crop == "Corn":
        df["GDU"] = (df["maxtemp"] + df["mintemp"]) / 2 - 10.0
        df["GDU"] = df["GDU"].clip(10, 30)
    elif crop == "Soy":
        df["GDU"] = (df["maxtemp"] + df["mintemp"]) / 2 - 14.0
        # df["GDU"] = df["GDU"].clip(14, 40)  # TODO: Check why this is commented out!
    df = df[df["GDU"].notnull()]
    # TODO: Aren't GDUs recalculated later? This may be unnecessary here.

    return df


def calc_rain_score(day: int, one_field_precip_bool: pd.Series) -> int:
    """
    Calculates a score (1-4) of antecedent rain conditions.

    Parameters
    ----------
    day : int
        Days after planting. Used as index for precipitation boolean series
    one_field_precip_bool : pd.Series
        Boolean series of whether or not significant (currently >= 2mm) precipitation
        was recorded for a day.

    Returns
    -------
    nbc_rd : int
        antecedent rain conditions score.
        # TODO: Link to documentation
    """
    one_field_precip_bool = one_field_precip_bool.iloc[(day - 1) : (day + 3)].copy()
    four_day_sum = one_field_precip_bool.sum()
    three_day_sums = one_field_precip_bool.rolling(3).sum()
    two_day_sums = one_field_precip_bool.rolling(2).sum()

    if four_day_sum == 0:
        nbc_rd = 1
    elif four_day_sum == 1:
        nbc_rd = 2
    elif four_day_sum == 2:
        nbc_rd = 3
    elif (two_day_sums == 2).any():
        nbc_rd = 3
    elif (three_day_sums == 3).any():
        nbc_rd = 3
    else:
        nbc_rd = 4

    return nbc_rd


def r_stella(
    one_field_weather: pd.DataFrame,
    ip_t_cof: pd.DataFrame,
    p_t_cof: pd.DataFrame,
    rc_t_input: pd.DataFrame,
    dvs_8_input: pd.DataFrame,
    rc_a_input: pd.DataFrame,
    p_opt: int,
    inocp: int,
    rrlex_par: float,
    rc_opt_par: float,
    ip_opt: int,
    is_fungicide: bool = False,
    fungicide: pd.DataFrame = pd.DataFrame(),
    fungicide_residual: pd.DataFrame = pd.DataFrame(),
) -> Tuple[pd.DataFrame, int]:
    
    """
    Calculates disease severity from weather data and crop-specific tuning parameters.
    # TODO: Link to documentation.

    Parameters
    ----------
    one_field_weather : pd.DataFrame
        Daily weather dataset for a single field. Necessary columns:
            `Temperature`: Degrees C
            `Rain`: Boolean
            `precip`: mm
    ip_t_cof : pd.DataFrame
        Crop-specific lookup table, indexed on Temperature (degrees C)
    p_t_cof : pd.DataFrame
        Crop-specific lookup table, indexed on Temperature (degrees C)
    rc_t_input : pd.DataFrame
        Crop-specific lookup table, indexed on Temperature (degrees C)
    dvs_8_input : pd.DataFrame
        Crop specific lookup table, indexed on Cumulative GDUs
    rc_a_input : pd.DataFrame
        Crop-specific lookup table, indexed on DVS 8
    p_opt : int
    inocp : int
    rrlex_par: float
    rc_opt_par: float
    ip_opt: int
    is_fungicide : bool
        Whether or not fungicide was applied
    fungicide : pd.DataFrame
        columns: `spray_number`, `spray_moment`, `spray_eff`
    fungicide_residual : pd.DataFrame
        Crop-specific lookup table

    Returns
    -------
    results : pd.DataFrame
    day : int
        final day after planting of model
    """
    # Input variables
    day = 1
    one_field_weather = one_field_weather.set_index("Day", drop=True)
    total_days = len(one_field_weather)
    ri_series = pd.Series(np.zeros((total_days)))
    output_columns = [
        "Temp",
        "RRDD",
        "ip",
        "p",
        "RcT",
        "Rc_W",
        "GDU",
        "GDUsum",
        "DVS8",
        "RcA",
        "Rc",
        "I",
        "RI",
        "COFR",
        "RSEN",
        "L",
        "HSEN",
        "TOTSITES",
        "RLEX",
        "R",
        "RG",
        "CumuLeak",
        "DIS",
        "Sev",
        "RAUPC",
        "RDI",
        "RDL",
        "LeakL",
        "LeakI",
        "REM",
        "Agg",
        "RRG",
        "SITEmax",
        "RRSEN",
        "inocp",
        "RRLEX",
        "RcOpt",
        "p_opt",
        "ip_opt",
        "H",
        "LAT",
        "AUDPC",
        "RT",
        "RTinc",
    ]
    # First Day initialization.
    results_day: Dict[str, Union[float, int]] = dict()
    results_day["p_opt"] = p_opt
    results_day["ip_opt"] = ip_opt
    results_day["Temp"] = one_field_weather["Temperature"].iloc[0]
    results_day["ip"] = ip_opt * np.interp(
        results_day["Temp"], ip_t_cof[0], ip_t_cof[1]
    )
    results_day["RRDD"] = 0.0001
    results_day["I"] = results_day["ip"]
    results_day["p"] = p_opt / np.interp(results_day["Temp"], p_t_cof[0], p_t_cof[1])
    results_day["L"] = 0
    results_day["AUDPC"] = 0
    results_day["GDUsum"] = 0
    results_day["H"] = 2500000
    results_day["HSEN"] = 0
    results_day["LeakI"] = 0
    results_day["LeakL"] = 0
    results_day["R"] = 0
    results_day["LAT"] = 0

    # add ResSpray variables
    if is_fungicide:
        results_day["ResSpray"] = 0

    results_day["CumuLeak"] = results_day["LeakL"] + results_day["LeakI"]
    results_day["DIS"] = (
        results_day["R"] + results_day["I"] + results_day["CumuLeak"] + results_day["L"]
    )
    results_day["TOTSITES"] = (
        results_day["HSEN"] + results_day["H"] + results_day["DIS"]
    )
    results_day["Sev"] = results_day["DIS"] / results_day["TOTSITES"]
    results_day["DVS8"] = np.interp(
        results_day["GDUsum"], dvs_8_input[0], dvs_8_input[1]
    )
    results_day["RAUPC"] = results_day["Sev"] if results_day["DVS8"] < 7 else 0
    results_day["GDU"] = results_day["Temp"] - 14
    results_day["RTinc"] = results_day["GDU"]
    results_day["SITEmax"] = 10000000
    results_day["RRG"] = 0.173
    results_day["RG"] = (
        results_day["RRG"]
        * results_day["H"]
        * (1 - (results_day["TOTSITES"] / results_day["SITEmax"]))
    )
    results_day["Rc_W"] = calc_rain_score(
        day=1, one_field_precip_bool=one_field_weather["Rain"]
    )
    results_day["RRLEX"] = rrlex_par
    results_day["inocp"] = inocp
    results_day["RcOpt"] = rc_opt_par - results_day["RRLEX"]
    results_day["RcT"] = np.interp(results_day["Temp"], rc_t_input[0], rc_t_input[1])
    results_day["RcA"] = np.interp(results_day["DVS8"], rc_a_input[0], rc_a_input[1])

    def get_rc_res(day: int, fungicide: pd.DataFrame, residual: pd.DataFrame) -> float:
        """
        Calculate Fungicide effective residual?

        Parameters
        ----------
        day : int
            Days since planting
        fungicide : pd.DataFrame
            columns: `spray_number`, `spray_moment`, `spray_eff`
        fungicide_residual : pd.DataFrame
            Crop-specific lookup table

        Returns
        -------
        fungicide_efficacy_residual : float
        """
        fungicide_efficacy_residual = 1.0
        fungicide["V4"] = fungicide["spray_moment"] + 7
        flag = (day > fungicide["spray_moment"]) & (day <= fungicide["V4"])
        if flag.any():
            eff_res = fungicide["spray_eff"][flag]
            fungicide_efficacy_residual = residual * eff_res
        return fungicide_efficacy_residual

    if is_fungicide:
        results_day["Residual"] = results_day["ResSpray"]
        results_day["RcFCur"] = 1
        results_day["FungEffcCur"] = 1
        results_day["EffRes"] = 1
        results_day["FungEffecRes"] = get_rc_res(
            day, fungicide, results_day["Residual"]
        )
        results_day["RcRes"] = 1
        fung_prod = results_day["RcFCur"] * results_day["FungEffecRes"]
    else:
        fung_prod = 1
    results_day["Rc"] = (
        results_day["RcOpt"]
        * results_day["RcT"]
        * results_day["RcA"]
        * results_day["Rc_W"]
        * fung_prod
    )
    results_day["COFR"] = 1 - (results_day["DIS"] / (2 * results_day["H"]))
    results_day["Agg"] = 1
    ri_series.iloc[0] = (
        results_day["Rc"]
        * results_day["I"]
        * np.power(results_day["COFR"], results_day["Agg"])
        + 1
    )
    results_day["RRSEN"] = 0.002307
    results_day["RSEN"] = results_day["RRDD"] + results_day["RRSEN"] * results_day["H"]
    results_day["RLEX"] = results_day["RRLEX"] * results_day["I"] * results_day["COFR"]
    results_day["RDI"] = 0.000100004821661
    results_day["REM"] = 0.999948211789

    def get_spray(day: int, daily_rainfall: float, fungicide: pd.DataFrame) -> float:
        """
        Calculate flow residual?

        Parameters
        ----------
        day : int
            Days since planting
        daily_rainfall : float
            daily precipitation in mm
        fungicide : pd.DataFrame
            columns: `spray_number`, `spray_moment`, `spray_eff`

        Returns
        -------
        flow_residual : float
        """
        flow_residual = 0.0
        fungicide["V4"] = fungicide["spray_moment"] + 7
        flag = (day > fungicide["spray_moment"]) & (day <= fungicide["V4"])

        if flag.any():
            flow_residual = daily_rainfall
        return flow_residual

    if is_fungicide:
        daily_rainfall = one_field_weather["precip"].iloc[day - 1]
        results_day["FlowRes"] = get_spray(day, daily_rainfall, fungicide)

    results_day["RT"] = 0
    results_day["RDL"] = 0

    # These two parameters are cumulative or otherwise do not fit the row-by-row
    # construction of a pandas dataframe.
    results_list = [results_day]
    rt_count = np.array([0] * total_days)

    # Propagate through other days.
    # TODO: much of this can likley be combined with first day.
    for day in range(2, total_days - 2):
        results_day = results_day.copy()
        results_day["I"] += (
            results_day["RT"]
            + results_day["RLEX"]
            - results_day["REM"]
            - results_day["RDI"]
        )
        results_day["L"] += (
            ri_series.loc[day - 2] - results_day["RT"] - results_day["RDL"]
        )
        results_day["AUDPC"] += results_day["RAUPC"]
        results_day["GDUsum"] += results_day["RTinc"]
        if day > 140:
            # This cuts off the model 140 days after planting.
            for col in set(output_columns) - {"I", "L", "AUDPC", "GDUsum"}:
                results_day[col] = 0
            results_list.append(results_day)
            rt_count = rt_count[: len(results_list) + 1]
            break
        results_day["H"] += (
            results_day["RG"]
            - ri_series.iloc[day - 2]
            - results_day["RSEN"]
            - results_day["RLEX"]
        )
        results_day["HSEN"] += results_day["RSEN"]
        results_day["LeakI"] += results_day["RDI"]
        results_day["LeakL"] += results_day["RDL"]
        results_day["R"] += results_day["REM"]

        if is_fungicide:
            results_day["ResSpray"] += results_day["FlowRes"]
        results_day["Temp"] = one_field_weather["Temperature"].iloc[day - 1]
        results_day["ip"] = ip_opt * np.interp(
            results_day["Temp"], ip_t_cof[0], ip_t_cof[1]
        )
        results_day["p"] = p_opt / np.interp(
            results_day["Temp"], p_t_cof[0].values, p_t_cof[1].values
        )
        results_day["CumuLeak"] = results_day["LeakL"] + results_day["LeakI"]
        results_day["DIS"] = (
            results_day["R"]
            + results_day["I"]
            + results_day["CumuLeak"]
            + results_day["L"]
        )
        results_day["TOTSITES"] = (
            results_day["HSEN"] + results_day["H"] + results_day["DIS"]
        )
        results_day["Sev"] = results_day["DIS"] / results_day["TOTSITES"]
        results_day["DVS8"] = np.interp(
            results_day["GDUsum"], dvs_8_input[0], dvs_8_input[1]
        )
        results_day["RAUPC"] = results_day["Sev"] if results_day["DVS8"] < 7 else 0
        results_day["GDU"] = results_day["Temp"] - 14
        results_day["RTinc"] = results_day["GDU"]
        results_day["RG"] = (
            results_day["RRG"]
            * results_day["H"]
            * (1 - (results_day["TOTSITES"] / results_day["SITEmax"]))
        )
        results_day["Rc_W"] = calc_rain_score(
            day=day, one_field_precip_bool=one_field_weather["Rain"]
        )
        results_day["RcT"] = np.interp(
            results_day["Temp"], rc_t_input[0], rc_t_input[1]
        )
        results_day["RcA"] = np.interp(
            results_day["DVS8"], rc_a_input[0], rc_a_input[1]
        )

        if is_fungicide:
            try:
                results_day["FungEffcCur"] = fungicide[
                    (fungicide["spray_moment"] <= day) & (day <= fungicide["V4"])
                ]["spray_eff"].iloc[0]
            except IndexError:
                results_day["FungEffcCur"] = 1
            results_day["RcFCur"] = (
                results_day["FungEffcCur"]
                if pd.notnull(results_day["FungEffcCur"])
                else 1
            )
            results_day["Residual"] = np.interp(
                results_day["ResSpray"], fungicide_residual[0], fungicide_residual[1]
            )
            results_day["RcRes"] = results_day["FungEffcCur"] * results_day["Residual"]
            results_day["FungEffecRes"] = (
                results_day["RcRes"] if pd.notnull(results_day["FungEffcCur"]) else 1
            )
            results_day["EffRes"] = results_day["FungEffcCur"]
            fung_prod = results_day["RcFCur"] * results_day["FungEffecRes"]
        else:
            fung_prod = 1
        results_day["Rc"] = (
            results_day["RcOpt"]
            * results_day["RcT"]
            * results_day["RcA"]
            * results_day["Rc_W"]
            * fung_prod
        )

        start = inocp if day > 10 else 0
        results_day["COFR"] = 1 - (
            results_day["DIS"] / (results_day["DIS"] + results_day["H"])
        )
        ri_series.iloc[day - 1] = (
            results_day["Rc"]
            * results_day["I"]
            * np.power(results_day["COFR"], results_day["Agg"])
            + start
        )
        results_day["RSEN"] = (
            results_day["RRDD"] + results_day["RRSEN"] * results_day["H"]
        )
        results_day["RLEX"] = (
            results_day["RRLEX"] * results_day["I"] * results_day["COFR"]
        )

        # ===================================================
        # Problems!!!!  # TODO: Problems!?!?
        results_day["RDI"] = results_day["I"] * results_day["RRDD"]
        results_day["RDL"] = results_day["L"] * results_day["RRDD"]
        if day > results_day["ip"]:
            # Note: math.floor(x) is not a direct replacement for math.ceil(x) - 1
            # where x is a whole number.
            results_day["REM"] = results_list[math.ceil(day - results_day["ip"]) - 1][
                "RT"
            ]
        # ===================================================
        if is_fungicide:
            daily_rainfall = one_field_weather["precip"].iloc[day - 1]
            results_day["FlowRes"] = get_spray(day, daily_rainfall, fungicide)

        rt_count[day - 1] = round(results_day["p"])
        results_day["RT"] = ri_series[rt_count == 0].sum()

        # Counting
        rt_count -= 1
        results_list.append(results_day)
    results = pd.DataFrame.from_dict(results_list)
    results["RI"] = ri_series
    # Match R's case sorted for comparison.
    # results = results[sorted(results.columns, key=str.casefold)]
    return results, day


def run_locationId_r_stella(
    all_fields_weather: pd.DataFrame,
    date: str,
    ip_t_cof: pd.DataFrame,
    p_t_cof: pd.DataFrame,
    rc_t_input: pd.DataFrame,
    rc_a_input: pd.DataFrame,
    dvs_8_input: pd.DataFrame,
    p_opt: int,
    inocp: int,
    rrlex_par: float,
    rc_opt_par: float,
    ip_opt: int,
    is_fungicide: bool = False,
    fungicide: Optional[pd.DataFrame] = None,
    fungicide_residual: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Calculates disease severity from weather data and crop-specific tuning parameters.
    # TODO: Link to documentation.

    Parameters
    ----------
    all_fields_weather : pd.DataFrame
        Daily weather dataset for all fields. Necessary columns:
            `locationId`: field identifier
            `DOY`: Julian day.
            `date`: date of weather record
            `maxtemp`: Degrees C
            `mintemp`: Degrees C
            `precip`: mm
    ip_t_cof : pd.DataFrame
        Crop-specific lookup table, indexed on Temperature (degrees C)
    p_t_cof : pd.DataFrame
        Crop-specific lookup table, indexed on Temperature (degrees C)
    rc_t_input : pd.DataFrame
        Crop-specific lookup table, indexed on Temperature (degrees C)
    rc_a_input : pd.DataFrame
        Crop-specific lookup table, indexed on DVS 8
    dvs_8_input : pd.DataFrame
        Crop specific lookup table, indexed on Cumulative GDUs
    p_opt : int
    inocp : int
    rrlex_par: float
    rc_opt_par: float
    ip_opt: int
    is_fungicide : bool
        Whether or not fungicide was applied
    fungicide : pd.DataFrame
        columns: `spray_number`, `spray_moment`, `spray_eff`
    fungicide_residual : pd.DataFrame
        Crop-specific lookup table

    Returns
    -------
    results : pd.DataFrame
        aggregated results by field.
    """
    all_fields_weather["Temperature"] = all_fields_weather[["maxtemp", "mintemp"]].mean(
        axis=1
    )
    all_fields_weather["Rain"] = (
        all_fields_weather["precip"] >= 2
    )  # Set rain as boolean
    result_list = []
    for i, (locale, one_field_weather) in enumerate(
        all_fields_weather.groupby("locationId")
    ):
        # print(f"Running location {i} Id = {locale}")
        one_field_weather = one_field_weather[one_field_weather["date"] >= date].copy()
        # Next iteration if date is not related in input file
        if len(one_field_weather) == 0:
            print(f"Location {i} Id = {locale}  NOT USED. No dates in range.")
            continue
        one_field_weather["Day"] = (
            one_field_weather["DOY"] - one_field_weather["DOY"].iloc[0]
        ).dt.days + 1
        field_results, n_day = r_stella(
            one_field_weather=one_field_weather,
            ip_t_cof=ip_t_cof,
            p_t_cof=p_t_cof,
            rc_t_input=rc_t_input,
            dvs_8_input=dvs_8_input,
            rc_a_input=rc_a_input,
            p_opt=p_opt,
            inocp=inocp,
            rrlex_par=rrlex_par,
            rc_opt_par=rc_opt_par,
            ip_opt=ip_opt,
            is_fungicide=is_fungicide,
            fungicide=fungicide,
            fungicide_residual=fungicide_residual,
        )
        # Output information
        start_date = one_field_weather["date"].iloc[0]
        end_date = one_field_weather["date"].iloc[n_day - 1]
        result_location = {}
        result_location["locationId"] = locale
        result_location["Date1"] = start_date
        result_location["Date2"] = end_date
        result_location["N_Days"] = (
            pd.to_datetime(end_date) - pd.to_datetime(start_date)
        ).days
        result_location["latitude"] = one_field_weather["latitude"].iloc[0]
        result_location["longitude"] = one_field_weather["longitude"].iloc[0]
        result_location["Sev50%"] = field_results["Sev"].median()
        result_location["SevMAX"] = field_results["Sev"].max()
        nonzero_sev = field_results[field_results["Sev"] != 0]["Sev"]
        if len(nonzero_sev):
            # note: The R version of this code was using `auc`,
            # but bypassing the `roc` calculation. `auc` seems to use a
            # trapezoidal area estimation.
            result_location["AUC"] = np.trapz(nonzero_sev)
        else:
            result_location["AUC"] = 0
        result_list.append(result_location)
    results = pd.DataFrame.from_dict(result_list)
    return results
