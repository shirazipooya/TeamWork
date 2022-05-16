# -----------------------------------------------------------------------------
# Corp Parameters
# -----------------------------------------------------------------------------

from typing import Dict, Optional, Tuple, Union, List

import numpy as np
import pandas as pd

CROP_PARAMETERS = {
    "Corn" : {
        
        'ip_t_cof': pd.DataFrame(
            {
                0: [10.00, 13.00, 15.50, 17.00, 20.00, 26.00, 30.00, 35.00],
                1: [0.00, 0.14, 0.27, 0.82, 1.00, 0.92, 0.41, 0.00]
            }
        ),
        
        'p_t_cof': pd.DataFrame(
            {
                0: [15.00, 20.00, 25.00],
                1: [0.60, 0.81, 1.00]
            }
        ),
        
        'rc_t_input': pd.DataFrame(
            {
                0: [15.00, 20.00, 22.50, 24.00, 26.00, 30.00],
                1: [0.22, 1.00, 0.44, 0.43, 0.41, 0.22]
            }
        ),
        
        'dvs_8_input': pd.DataFrame(
            {
                0: [110.00, 200.00, 350.00, 475.00, 610.00, 740.00, 1135.00, 1660.00, 1925.00, 2320.00, 2700.00],
                1: [0.00, 1.00, 2.00, 3.00, 4.00, 5.00, 6.00, 8.00, 9.00, 10.00, 11.00]
            }
        ),
        
        'rc_a_input': pd.DataFrame(
            {
                0: [0.00, 1.00, 2.00, 3.00, 4.00, 5.00, 6.00, 7.00],
                1: [1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00]
            }
        ),
        
        'fungicide': pd.DataFrame(
            {
                0: [1, 2, 3],
                1: [45, 62, 79],
                2: [0.4, 0.4, 0.4]
            }
        ),
        
        'fungicide_residual': pd.DataFrame(
            {
                0: [0.00, 5.00, 10.00, 15.00, 20.00],
                1: [1.00, 0.80, 0.50, 0.25, 0.00]
            }
        )        
    },

    "Soy" : {
        
        'ip_t_cof': pd.DataFrame(
            {
                0: [5.00, 11.00, 17.00, 23.00, 29.00, 35.00],
                1: [0.00, 0.33, 0.60, 1.00, 1.00, 0.00]
            }
        ),
        
        'p_t_cof': pd.DataFrame(
            {
                0: [0.00, 4.00, 8.00, 12.00, 16.00, 20.00, 24.00, 28.00, 32.00, 36.00, 40.00],
                1: [0.00, 0.31, 0.39, 0.53, 0.82, 1.00, 1.00, 0.75, 0.60, 0.30, 0.00]
            }
        ),
        
        'rc_t_input': pd.DataFrame(
            {
                0: [10.00, 12.50, 15.00, 17.50, 20.00, 23.00, 25.00, 27.50, 30.00],
                1: [0.00, 0.57, 0.88, 1.00, 1.00, 0.86, 0.61, 0.41, 0.00]
            }
        ),
        
        'dvs_8_input': pd.DataFrame(
            {
                0: [137.00, 366.00, 595.00, 824.00, 1053.00, 1282.00, 1511.00, 1740.00],
                1: [0.00, 1.00, 2.00, 3.00, 4.00, 5.00, 6.00, 7.00]
            }
        ),
        
        'rc_a_input': pd.DataFrame(
            {
                0: [0.00, 1.00, 2.00, 3.00, 4.00, 5.00, 6.00, 7.00],
                1: [1.0000, 1.0000, 1.0000, 0.1000, 0.0001, 0.0001, 0.0001, 0.0001]
            }
        ),
        
        'fungicide': pd.DataFrame(
            {
                0: [1, 2, 3],
                1: [45, 62, 79],
                2: [0.4, 0.4, 0.4]
            }
        ),
        
        'fungicide_residual': pd.DataFrame(
            {
                0: [0.00, 5.00, 10.00, 15.00, 20.00],
                1: [1.00, 0.80, 0.50, 0.25, 0.00]
            }
        )        
    }  
}


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