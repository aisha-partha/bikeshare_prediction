import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

from bike_model.config.core import config
from bike_model.processing.features import *

bike_pipe = Pipeline([

    ## Weekday imputer ##
    ('weekday_imputation', WeekdayImputer(col_name=config.modl_config.weekday_var)), 

    ## Weather imputer ##
    ('weather_imputation', WeathersitImputer(col_name=config.modl_config.weathersit_var)),

    ## Unused columns ##
    ('unused_column_dropper',ColumnDropper(col_list=config.modl_config.cols_delete)),

    ## Mapping ##
    ('all_mapper', Mapper(col_map=config.modl_config.mapping_dict)),

    ## Outlier handling ##
    ('all_outlier', OutlierHandler(col_list=config.modl_config.num_cols)),

    ## One hot encoder ##
    ('one_hot_encoder', WeekdayOneHotEncoder(col_list=config.modl_config.onehot_cols)),

    ## scaler ##
    ('scaler', StandardScaler()),

    ## Model fit
    ('model_rf', RandomForestRegressor(n_estimators=config.modl_config.n_estimators, 
                                       max_depth=config.modl_config.max_depth, 
                                       random_state=config.modl_config.random_state))
])