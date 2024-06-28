
"""
Note: These tests will fail if you have not first trained the model.
"""
import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

import numpy as np
from bike_model.config.core import config
from bike_model.processing.features import *
from bike_model.processing.data_manager import pre_pipeline_preparation


def test_weekday_variable_transformer(sample_input_data):
    
    df_test = sample_input_data[0].copy()
    # Given
    transformer = WeekdayImputer(
        col_name=config.modl_config.weekday_var 
    )
    assert np.isnan(df_test.loc[7046,'weekday'])

    # When
    subject = transformer.fit(df_test).transform(df_test)

    # Then
    assert subject.loc[7046,'weekday'] == 'Wed'
    

def test_weathersit_variable_transformer(sample_input_data):
    df_test = sample_input_data[0].copy()
    # Given
    transformer = WeathersitImputer(
        col_name=config.modl_config.weathersit_var
    )
    assert np.isnan(df_test.loc[6147,'weathersit'])

    # When
    subject = transformer.fit(df_test).transform(df_test)

    # Then
    assert subject.loc[6147,'weathersit'] == 'Clear'


def test_columndrop_variable_transformer(sample_input_data):
    df_test = sample_input_data[0].copy()
    # Given
    transformer = ColumnDropper(
        col_list=config.modl_config.cols_delete
    )
    
    assert all([True for col in config.modl_config.cols_delete if col in df_test.columns])==True

    # When
    subject = transformer.fit(df_test).transform(df_test)

    # Then
    assert all([True for col in config.modl_config.cols_delete if not col in subject.columns])==True
    

def test_mapper_variable_transformer(sample_input_data):
    df_test = sample_input_data[0].copy()
    df_prepped = pre_pipeline_preparation(data_frame=df_test)
    # Given
    transformer = Mapper(
        col_map=config.modl_config.mapping_dict
    )
    
    for key, val in config.modl_config.mapping_dict.items():
        
        assert all([True for i in list(val.keys()) if i in df_prepped[key]])

        # When
        subject = transformer.fit(df_prepped).transform(df_prepped)

        # Then
        assert all([True for i in list(val.values()) if i in subject[key]])

  
def test_outlier_variable_transformer(sample_input_data):
    df_test = sample_input_data[0].copy()
    # Given
    transformer = OutlierHandler(
        col_list=config.modl_config.num_cols
    )
    
    for col in config.modl_config.num_cols:
        
        q1 = df_test.describe()[col].loc['25%']
        q3 = df_test.describe()[col].loc['75%']
        iqr = q3 - q1
        lower_bound = q1 - (1.5 * iqr)
        upper_bound = q3 + (1.5 * iqr)
        
        assert len(df_test[df_test[col] > upper_bound]) + len(df_test[df_test[col] < lower_bound]) >= 0

        # When
        subject = transformer.fit(df_test).transform(df_test)

        # Then
        assert (len(subject[subject[col] > upper_bound]) + len(subject[subject[col] < lower_bound])) == 0


def test_onehot_variable_transformer(sample_input_data):
    df_test = sample_input_data[0].copy()
    # Given
    
    transformer1 = WeekdayImputer(
        col_name=config.modl_config.weekday_var 
    )
    
    subject1 = transformer1.fit(df_test).transform(df_test)
    
    transformer2 = WeekdayOneHotEncoder(
        col_list=config.modl_config.onehot_cols
    )
    
    for col in config.modl_config.onehot_cols:
        
        assert all([True for day in transformer2.categories_ if day in df_test[col]])==True

        # When
        subject2 = transformer2.fit(df_test).transform(df_test)

        # Then
        assert sum(subject2[subject2.columns[12:]].iloc[0, :])==1
        assert len(subject2.columns) == 19