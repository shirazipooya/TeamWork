# -----------------------------------------------------------------------------
# Read Corp Parameters
# -----------------------------------------------------------------------------

from typing import Dict, Optional, Tuple, Union, List

import numpy as np
import pandas as pd


def corp_parameters(
    crop_mechanistic: List[str] = None,
    crop_parameters_path: List[str] = None,
) -> Dict:
    
    para = ["ip_t_cof", "p_t_cof", "rc_t_input", "dvs_8_input", "rc_a_input", "fungicide", "fungicide_residual"]
      
    crop_para = dict()
    
    if (crop_parameters_path is not None) and (crop_mechanistic is not None):
        if len(crop_mechanistic) == len(crop_parameters_path):
            for i in range(len(crop_mechanistic)):
                crop_para[crop_mechanistic[i]] = dict()
                for p in para:
                    crop_para[crop_mechanistic[i]][p] = pd.read_excel(crop_parameters_path[i], engine="openpyxl", sheet_name=p, header=None)

    return crop_para