# -----------------------------------------------------------------------------
# Calculate Fungicide Effective Residual
# -----------------------------------------------------------------------------

from typing import Dict, Optional, Tuple, Union, List

import numpy as np
import pandas as pd

def get_rc_res(
    day: int, 
    fungicide: pd.DataFrame, 
    residual: pd.DataFrame
):
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