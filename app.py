from flask import Flask,request,render_template
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData,PredictPipeline

application = Flask(__name__)

app = application

## Route for a home page 

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata',methods=['GET','POST'])

def predict_datapoint():
    if request.method =='GET':
        return render_template('home.html')

    else:
        data = CustomData(
            Age=request.form.get("Age"),
            Sex=request.form.get("Sex"),
            ALB=request.form.get("ALB"),
            ALP=request.form.get("ALP"),
            ALT=request.form.get("ALT"),
            AST=request.form.get("AST"),
            BIL=request.form.get("BIL"),
            CHE=request.form.get("CHE"),
            CHOL=request.form.get("CHOL"),
            CREA=request.form.get("CREA"),
            GGT=request.form.get("GGT"),
            PROT=request.form.get("PROT")
        
        
        )
        
        pred_df=data.get_data_as_data_frame()
        print(pred_df)
        print("Before Prediction")

        predict_pipeline=PredictPipeline()
        results=predict_pipeline.predict(pred_df)
        print("after Prediction")
        return render_template('home.html',results=int(results[0]))
    

if __name__=="__main__":
    app.run(host="0.0.0.0", debug =True)