"""
Note: These tests will fail if you have not first trained the model.
"""

from bikeshare_model.config.core import config
from bikeshare_model.processing.features import WeekdayImputer
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

import sys
from pathlib import Path

file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

'''   
def test_age_variable_transformer(sample_input_data):
    # Given
    transformer = age_col_tfr(
        variables=config.model_config.age_var,  # cabin
    )
    assert np.isnan(sample_input_data[0].loc[709, 'Age'])

    # When
    subject = transformer.fit(sample_input_data[0]).transform(sample_input_data[0])

    # Then
    assert subject.loc[709, 'Age'] == 21


'''

def test_WeekdayImputer(sample_input_data):
    
    ##print ("*************************")
    ##print(sample_input_data[0][config.model_config.weekday_var])
    #print ("*************************")
    #assert pd.isnull(bikeshare['weekday']).any()
    assert pd.isnull(sample_input_data[0][config.model_config.weekday_var]).any()
    
    # Creating the pipeline
    pipeline = Pipeline([
        ('weekday_imputer', WeekdayImputer(feature='weekday'))
    ])
    
    #cwt = WeekdayImputer(feature=config.model_config.weekday_var)
    #cwt = WeekdayImputer(feature='weekday')
    #data1 = cwt.fit_transform(sample_input_data[0])
    ##data1 = cwt.fit_transform(sample_input_data[0])
    #data2 = data1.transform(sample_input_data[0])
    
    data_transformed = pipeline.fit_transform(sample_input_data[0])
    
    
    print ("*************************")
    assert not pd.isnull(data_transformed['weekday']).any()
    print ("*************************")
    #assert not pd.isnull(data1['weekday']).any()
    #assert not np.isnan(data1['weekday']).any()

