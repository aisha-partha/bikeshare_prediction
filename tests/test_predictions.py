"""
Note: These tests will fail if you have not first trained the model.
"""

import sys
import numpy as np
#from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error, r2_score

from bikeshare_model.predict import make_prediction

def test_make_prediction(sample_input_data):
    # Given
    expected_no_predictions = 3476

    
    # When
    result = make_prediction(input_data=sample_input_data[0])

    print ("*************************")
    print(sample_input_data[0])
    print ("*************************")
    
    # Then
    predictions = result.get("predictions")
    
    print ("*************************")
    print(predictions)
    print ("*************************")
    
    assert isinstance(predictions, np.ndarray)
    assert isinstance(predictions[0], np.float64) 
    assert result.get("errors") is None
    assert len(predictions) == expected_no_predictions
    
    _predictions = list(predictions)
    y_true = sample_input_data[1]
    
    r2Score = r2_score(y_true,_predictions)
    print (r2Score)
    #r2_score(y_test, y_pred)
    #assert accuracy > 0.8
    assert r2Score > 0.92
