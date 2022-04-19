
# -----------------------------------------------------------------------------
# Calculates a Score (1-4) of Antecedent Rain Conditions
# -----------------------------------------------------------------------------

from typing import Dict, Optional, Tuple, Union, List

import numpy as np
import pandas as pd

def calc_rain_score(
    day: int,
    one_field_precip_bool: pd.Series
):
    
    """
    Calculates a Score (1-4) of Antecedent Rain Conditions.

    Args:
        day (int): Days After Planting. Used as index for Precipitation Boolean Series.
        
        one_field_precip_bool (pd.Series): Boolean Series of Whether or not Significant (currently >= 2mm) Precipitation Was Recorded for a Day.

    Returns:
        nbc_rd (int): Antecedent Rain Conditions Score.
    """
    
    one_field_precip_bool = one_field_precip_bool.iloc[(day - 1):(day + 3)].copy()
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