import os
import sys

import numpy as np 
import pandas as pd 
import dill
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)


    except Exception as e:
        raise CustomException(e, sys)
    
def evaluate_model(X_train, y_train, X_test,  y_test, models,param):
    """
    Fit and evaluate given machine learning models.
    model : a dict different sklearn ml model
    X_train : trainig data (no labels)
    y_train : training labels
    X_text : testing data (no labels)
    y_test : testing labels
    param : param grid for hyperparameter of different model 
    
    """
    try:
        # set random seed 
        np.random.seed(42)
        # Make a dictionary to keep model scores
        report = {}
        # Loop through models 
        for i in range(len(list(models))):
            model = list(models.values())[i]
            para=param[list(models.keys())[i]]

            gs = GridSearchCV(model,para,cv=3)
            gs.fit(X_train,y_train)



            #Fit the model to the data
            model.set_params(**gs.best_params_)
            model.fit(X_train, y_train)

            # Evaluate the model 
            y_train_pred = model.predict(X_train)

            y_test_pred = model.predict(X_test)

            train_model_score = r2_score(y_train, y_train_pred)

            test_model_score = r2_score(y_test, y_test_pred)

            report[list(models.keys())[i]] = test_model_score

        return report 

    except Exception as e:
            raise CustomException(e, sys)

    