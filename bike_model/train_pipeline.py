import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

from bike_model.config.core import config
from bike_model.pipeline import bike_pipe
from bike_model.processing.data_manager import load_dataset, save_pipeline

def run_training() -> None:
    
    """
    Train the model.
    """

    # read training data
    data = load_dataset(file_name=config.app_config.training_data_file)

    # divide train and test
    X_train, X_test, y_train, y_test = train_test_split(
        data[config.modl_config.features],  # predictors
        data[config.modl_config.target],
        test_size=config.modl_config.test_size,
        # we are setting the random seed here
        # for reproducibility
        random_state=config.modl_config.random_state,
    )

    # Pipeline fitting
    bike_pipe.fit(X_train,y_train)
    y_pred = bike_pipe.predict(X_test)
    print("R2 score:", round(r2_score(y_test, y_pred),2))
    print("Mean squared error:", round(mean_squared_error(y_test, y_pred),2))

    # persist trained model
    save_pipeline(pipeline_to_persist= bike_pipe)
    # printing the score
    
if __name__ == "__main__":
    run_training()