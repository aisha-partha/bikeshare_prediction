
"""
Note: These tests will fail if you have not first trained the model.
"""
import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

import numpy as np

from bikeshare_model.config.core import config
from bikeshare_model.processing.features import *
from bikeshare_model.processing.data_manager import pre_pipeline_preparation



def test_weathersit_variable_transformer(sample_input_data):
    # Given
    test_df = sample_input_data[0].copy()
    transformer = WeathersitImputer(
        feature=config.model_config.weathersit_var,  
    )
    
    assert pd.isnull(test_df.loc[101,config.model_config.weathersit_var])

    # When
    subject = transformer.fit(test_df).transform(test_df)

    # Then
    assert subject.loc[101,config.model_config.weathersit_var] == 'Clear'
    
def test_weekdayimputer_variable_transformer(sample_input_data):
    # Given
    test_df = sample_input_data[0].copy()
    transformer = WeekdayImputer(
        feature=config.model_config.weekday_var,  
    )
    
    assert pd.isnull(test_df.loc[424,config.model_config.weekday_var])
    
    # When
    subject = transformer.fit(test_df).transform(test_df)

    # Then
    assert subject.loc[424,config.model_config.weekday_var] == 'Sun'
    

def test_outlier_transformer(sample_input_data):
    # Given
    test_df = sample_input_data[0].copy()
    transformer = OutlierHandler(
        feature=config.model_config.windspeed_var,  
    )
    
    Q1 = np.percentile(test_df.loc[:, config.model_config.windspeed_var], 25)
    Q3 = np.percentile(test_df.loc[:, config.model_config.windspeed_var], 75)
    deviation_allowed = 1.5*(Q3 - Q1)
    lower_bound = Q1 - deviation_allowed
    upper_bound = Q3 + deviation_allowed

    #Given
    assert len(test_df[test_df[config.model_config.windspeed_var] > upper_bound]) >= 0
    
    # When
    subject = transformer.fit(test_df).transform(test_df)

    # Then
    assert len(subject[subject[config.model_config.windspeed_var] > upper_bound]) == 0
    

def test_columndropper_transformer(sample_input_data):
    # Given
    test_df = sample_input_data[0].copy()
    transformer = ColumnDropperTransformer(
        columns=config.model_config.unused_fields,  
    )
    
    assert len(test_df.columns) == 13
    
    
    # When
    subject = transformer.fit(test_df).transform(test_df)

    # Then
    assert len(subject.columns) == 10
    
    
    
def test_mapper_season(sample_input_data):
    # Given
    test_df = sample_input_data[0].copy()
    transformer = Mapper(
        config.model_config.season_var,  config.model_config.season_mappings
    )
    
    assert set(test_df['season'].unique()) == {'winter', 'fall', 'spring','summer'}

    # When
    subject = transformer.fit(test_df).transform(test_df)

    # Then
    assert set(subject['season'].unique()) == {0,1,2,3} 
    
    
def test_mapper_year(sample_input_data):
    # Given
    test_df = sample_input_data[0].copy()
    transformed_df = pre_pipeline_preparation(data_frame=test_df)
    transformer = Mapper(
        config.model_config.yr_var,  config.model_config.yr_mappings
    )
    
    assert set(transformed_df['yr'].unique()) == {'2011', '2012'}

    # When
    subject = transformer.fit(transformed_df).transform(transformed_df)

    # Then
    assert set(subject['yr'].unique()) == {0,1} 
    

def test_weekday_ohe(sample_input_data):
    # Given
    test_df = sample_input_data[0].copy()

    transformer1 = WeekdayImputer(
        feature=config.model_config.weekday_var,  
    )
    transformer2 = WeekdayOneHotEncoder(
        config.model_config.weekday_var
    )
    
    subject_1 = transformer1.fit(test_df).transform(test_df)
    assert list(subject_1[config.model_config.weekday_var].unique()).sort() == ['Sun', 'Sat', 'Mon', 'Tue','Wed', 'Thu', 'Fri'].sort()
    assert len(subject_1.columns) == 13
    # When
    
    subject_2 = transformer2.fit(subject_1).transform(subject_1)

    # Then
    assert len(subject_2.columns) == 19
    #assert set(subject['yr'].unique()) == {0,1,2,3,4,5,6} 
    
