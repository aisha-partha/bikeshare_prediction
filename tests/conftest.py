import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

import pytest
import pandas as pd
from sklearn.model_selection import train_test_split

from bike_model.config.core import config
from bike_model.processing.data_manager import _load_raw_dataset



@pytest.fixture
def sample_input_data():
    data = _load_raw_dataset(file_name=config.app_config.training_data_file)
    data['dteday'] = pd.to_datetime(data['dteday'], format='%Y-%m-%d')
    # divide train and test
    X_train, X_test, y_train, y_test = train_test_split(
        data.drop(config.modl_config.target, axis=1),  # predictors
        data[config.modl_config.target],
        test_size=config.modl_config.test_size,
        # we are setting the random seed here
        # for reproducibility
        random_state=config.modl_config.random_state,
    )
    
    return [X_test, y_test]