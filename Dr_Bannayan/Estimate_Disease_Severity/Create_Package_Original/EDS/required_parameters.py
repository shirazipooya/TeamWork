# -----------------------------------------------------------------------------
# Required Parameters
# -----------------------------------------------------------------------------

from typing import Dict, Optional, Tuple, Union, List

import numpy as np
import pandas as pd


def required_parameters(
    p_opt: List[float] = [7, 10, 14],
    rc_opt_par: List[float] = [0.35, 0.25, 0.15],
    rrlex_par: List[float] = [0.1, 0.01, .0001],
    spray_number: List[int] = [1, 2, 3],
    spray_moment: List[int] = [30, 45, 60],
    spray_eff: List[float] = [0.5, 0.5, 0.5]
) -> dict:
      
    para = dict()
    
    para["fungicide_inputs_full"] = pd.DataFrame(
        {
            "spray_number": spray_number,
            "spray_moment": spray_moment,
            "spray_eff": spray_eff
        }
    )
    
    para["genetic_mechanistic_para"] = {
        
        "Susceptible" : {
            "p_opt" : p_opt[0],
            "rc_opt_par" : rc_opt_par[0],
            "rrlex_par" : rrlex_par[0]
        },
        
        "Moderate" : {
            "p_opt" : p_opt[1],
            "rc_opt_par" : rc_opt_par[1],
            "rrlex_par" : rrlex_par[1]
        },
        
        "Resistant" : {
            "p_opt" : p_opt[2],
            "rc_opt_par" : rc_opt_par[2],
            "rrlex_par" : rrlex_par[2]
        }
        
    }
     
    return para