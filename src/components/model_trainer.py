import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass
import os
import sys


# Models from Scikit-learn
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor

# Model Evaluations 
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, cross_val_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import precision_score, recall_score, f1_score,r2_score 

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object,evaluate_model

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts', "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()

    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Split training and test input data")
            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            models = {
                "Random Forest Classifier": RandomForestClassifier(),
                "Random Forest Regressor": RandomForestRegressor(),
                "K-Neighbors Classifier": KNeighborsClassifier(),
                "Logistic Regression": LogisticRegression()
            }

            params = {
                
                "Random Forest Classifier":{
                    'n_estimators': [100, 200, 300],          # Number of trees
                    'max_features': ['sqrt'],
                    'max_depth': [None, 5, 10, 15],            # Maximum depth of trees
                    #'min_samples_leaf': [1, 2, 4],             # Minimum number of samples required at each leaf node
                },
                "Random Forest Regressor":{
                    'n_estimators': [100, 200, 300],          # Number of trees
                    'max_depth': [None, 5, 10, 15],            # Maximum depth of trees
                    'max_features': ['sqrt']
                    #'min_samples_split': [2, 5, 10],           # Minimum number of samples required to split a node
                    #'min_samples_leaf': [1, 2, 4],             # Minimum number of samples required at each leaf node

                },
                "K-Neighbors Classifier":{
                    'n_neighbors': [3, 5, 7],             # Number of neighbors to consider
                    'weights': ['uniform', 'distance'],   # Weight function used in prediction
                    #'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],   # Algorithm used to compute nearest neighbors
                    'leaf_size': [20, 30, 40],             # Leaf size passed to BallTree or KDTree
                },
                "Logistic Regression":{
                    'C': [0.1, 1.0, 10.0],                   # Inverse of regularization strength
                    'solver': ['liblinear']          # Algorithm to use in the optimization problem

                }
                
            }


            model_report:dict=evaluate_model(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,models=models,param=params)

            ## To get the best model score from dict
            best_model_score = max(sorted(model_report.values()))

            ## To get best model name from dict

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]

            if best_model_score<0.5:
                raise CustomException("No best model found")
            
            logging.info(f"Best found model on both training and testing dataset")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted = best_model.predict(X_test)

            r2_square = r2_score(y_test, predicted)

            return r2_square



        except Exception as e:
            raise CustomException(e,sys)