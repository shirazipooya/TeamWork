# -----------------------------------------------------------------------------
# Calculate Flow Residual
# -----------------------------------------------------------------------------

from typing import Dict, Optional, Tuple, Union, List

import numpy as np
import pandas as pd

def get_spray(
    day: int, 
    daily_rainfall: float, 
    fungicide: pd.DataFrame
) -> float:
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