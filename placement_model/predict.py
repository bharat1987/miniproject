import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

from typing import Union
import pandas as pd
import numpy as np
from placement_model import __version__ as _version
from placement_model.config.core import config
from placement_model.pipeline import placement_pipe
from placement_model.processing.data_manager import load_pipeline
from placement_model.processing.data_manager import pre_pipeline_preparation
from placement_model.processing.validation import validate_inputs

pipeline_file_name = f"{config.app_config.pipeline_save_file}{_version}.pkl"
placement_pipe= load_pipeline(file_name=pipeline_file_name)

def make_prediction(*,input_data:Union[pd.DataFrame, dict]) -> dict:
    """Make a prediction using a saved model """

    validated_data, errors = validate_inputs(input_df=pd.DataFrame(input_data))
    
    validated_data=validated_data.reindex(columns=config.model_config.features)
    results = {"predictions": None, "version": _version, "errors": errors}
    
    predictions = placement_pipe.predict(validated_data)

    results = {"predictions": predictions.tolist(),"version": _version, "errors": ''}
    print(results)
    if not errors:

        predictions = placement_pipe.predict(validated_data)
        results = {"predictions": predictions.tolist(),"version": _version, "errors": errors}
        print(results)
    return results

if __name__ == "__main__":

    data_in={'sl_no':[100],'gender':['M'],'ssc_p':[62.00],'ssc_b':["Central"],'hsc_p':[58.00],'hsc_b':["Others"],
                'hsc_s':["Science"],'degree_p':[53.00],'degree_t':['Comm&Mgmt'],'workex':["No"],'etest_p':[89],'specialisation':["Mkt&HR"],
                'mba_p':[60.22],'salary':[0]}
    
    make_prediction(input_data=data_in)
