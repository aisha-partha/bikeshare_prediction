"""
Note: These tests will fail if you have not first trained the model.
"""

from bikeshare_model.config.core import config
from bikeshare_model.processing.features import WeekdayImputer,Mapper,WeathersitImputer
from bikeshare_model.processing.data_manager import get_year_and_month,_load_raw_dataset,pre_pipeline_preparation
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

import sys
from pathlib import Path
import pytest, warnings

file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

data_transformed = None

def test_WeekdayImputer(sample_input_data):
    #assert pd.isnull(bikeshare['weekday']).any()
    assert pd.isnull(sample_input_data[0][config.model_config.weekday_var]).any()
    # Creating the pipeline
    pipeline = Pipeline([
        ('weekday_imputer', WeekdayImputer(feature='weekday'))
        #('weekday_imputer', WeekdayImputer(feature=config.model_config.weekday_var))
    ])
    global data_transformed
    data_transformed = pipeline.fit_transform(sample_input_data[0])
    assert not pd.isnull(data_transformed['weekday']).any()
    

def test_Mapping(sample_input_data):
    global data_transformed
    data1 = data_transformed

    # Print initial columns
    #print("Columns in data1 before any transformation:", data1.columns)

    # Check if 'yr' column is present in data1
    #assert 'yr' in data1.columns, "'yr' column is missing from the initial data1 DataFrame"
    
    pipeline = Pipeline([
        ('weatherisimp', WeathersitImputer(feature = config.model_config.weathersit_var))
        #('weathersit_imputer', WeathersitImputer(feature='weathersit'))
        #('weekday_imputer', WeekdayImputer(feature=config.model_config.weekday_var))
    ])
    
    
    

    #wsi = WeathersitImputer(feature='weathersit')
    data2 = pipeline.fit_transform(data1)

    # Print columns after transformation
    #print("Columns in data2 after weathersit imputation:", data2.columns)

    # Adjust column mappings according to data1 and data2 columns
    col_mapping = {
                    "season" : {'spring': 0, 'winter': 1, 'summer': 2, 'fall': 3},
                    "hr" : {'4am': 0, '3am': 1, '5am': 2, '2am': 3, '1am': 4, '12am': 5, '6am': 6, '11pm': 7, '10pm': 8, '10am': 9, '9pm': 10, '11am': 11, '7am': 12, '9am': 13, '8pm': 14, '2pm': 15, '1pm': 16, '12pm': 17, '3pm': 18, '4pm': 19, '7pm': 20, '8am': 21, '6pm': 22, '5pm': 23},
                    "holiday" : {'Yes': 1, 'No': 0},
                    "weekday" : {'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3, 'Friday': 4, 'Saturday': 5, 'Sunday': 6},
                    "workingday" : {'No': 0, 'Yes': 1},
                    "weathersit" : {'Heavy Rain': 0, 'Light Rain': 1, 'Mist': 2, 'Clear': 3}
                }

    # Verify data after weathersit imputer
    #print("Columns in data2 before applying column mappings:", data2.columns)
    
    data3 = data2.copy()
    for key, value in col_mapping.items():
        #print(key, value)
        col_mapper = Mapper(key, value)
        data3 = col_mapper.fit(data2).transform(data3)
    
    # Verify data after mapping
    print("Columns in data3 after applying column mappings:", data3.head)

    # Check if data2 and data3 are identical
    #test_dataframes_identical(data2, data3)
    #compare_dataframes(data2, data3)
    #assert data2['weathersit'].equals(data3['weathersit'])
    assert 1 in data3['weathersit'].values
    #assert data3['season']['winter']==1
    
    for original, mapped in col_mapping['season'].items():
        assert mapped in data3['season'].values
# Run the test case
#test_Mapping(sample_input_data)
