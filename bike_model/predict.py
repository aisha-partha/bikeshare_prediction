import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

from typing import Union
import pandas as pd
import numpy as np

from bike_model import __version__ as _version
from bike_model.config.core import config
from bike_model.pipeline import bike_pipe
from bike_model.processing.data_manager import load_pipeline
from bike_model.processing.data_manager import pre_pipeline_preparation
from bike_model.processing.validation import validate_inputs


pipeline_file_name = f"{config.app_config.pipeline_save_file}{_version}.pkl"
titanic_pipe= load_pipeline(file_name=pipeline_file_name)


def make_prediction(*,input_data:Union[pd.DataFrame, dict]) -> dict:
    """Make a prediction using a saved model """

    validated_data, errors = validate_inputs(input_df=pd.DataFrame(input_data))
    
    try:
        
        #validated_data=validated_data.reindex(columns=['Pclass','Sex','Age','Fare', 'Embarked','FamilySize','Has_cabin','Title'])
        validated_data=validated_data.reindex(columns=config.modl_config.features)
        #print(validated_data)
    
    except Exception as e:
        return e
    
    results = {"predictions": None, "version": _version, "errors": errors}
    
    predictions = titanic_pipe.predict(validated_data)

    results = {"predictions": predictions,"version": _version, "errors": errors}
    print(results)
    
    if not errors:

        predictions = titanic_pipe.predict(validated_data)
        results = {"predictions": predictions,"version": _version, "errors": errors}
        #print(results)

    return results

if __name__ == "__main__":

    data_in={'dteday': ['2011-09-26'],
             'season': ['winter'],
             'hr': ['7am'],
             'holiday': ['No'],
             'weekday': ['Mon'],
             'workingday': ['Yes'],
             'weathersit': ['Mist'],
             'temp': [21.14],
             'atemp': [20.003],
             'hum': [94],
             'windspeed': [0],
             'casual': [17],
             'registered': [315]
    }
    
    make_prediction(input_data=data_in)
