import sys
import pandas as pd 
from src.exception import CustomException
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass
    def predict(self,features):
        try:
            model_path = 'artifacts\model.pkl'
            preprocessor_path = 'artifacts\preprocessor.pkl'
            model=load_object(file_path=model_path)
            preprocessor=load_object(file_path=preprocessor_path)

            data_scaled = preprocessor.transform(features)
            print(data_scaled)
            preds=model.predict(data_scaled)
            return preds
        except Exception as e:
            raise CustomException(e, sys)


class CustomData:
    def __init__(self,
                 Age: int,
                 Sex: str,
                 ALB: float,
                 ALP: float,
                 ALT: float,
                 AST: float,
                 BIL: float,
                 CHE: float,
                 CHOL: float,
                 CREA: float,
                 GGT: float,
                 PROT: float):
        
        self.Age=Age
        self.Sex=Sex
        self.ALB=ALB
        self.ALP=ALP
        self.ALT=ALT
        self.AST=AST
        self.BIL=BIL
        self.CHE=CHE
        self.CHOL=CHOL
        self.CREA=CREA
        self.GGT=GGT
        self.PROT=PROT

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {

                "Age": [self.Age],
                "Sex": [self.Sex],
                "ALB": [self.ALB],
                "ALP": [self.ALP],
                "ALT": [self.ALT],
                "AST": [self.AST],
                "BIL": [self.BIL],
                "CHE": [self.CHE],
                "CHOL": [self.CHOL],
                "CREA": [self.CREA],
                "GGT":[self.GGT],
                "PROT": [self.PROT]

            }
            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e,sys)