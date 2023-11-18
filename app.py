from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
from pycaret.regression import *
import pandas as pd
import sys

app = FastAPI()


model = joblib.load("modelo_final.pkl")
le_sex = joblib.load('le_sex.pkl')
le_chestpaintype = joblib.load('le_chestpaintype.pkl')
le_restingecg = joblib.load('le_restingecg.pkl')
le_exerciseangina = joblib.load('le_exerciseangina.pkl')
le_stslope = joblib.load('le_stslope.pkl')

sex_mapping = dict(zip(le_sex.classes_, le_sex.transform(le_sex.classes_)))
print(sex_mapping)
class InputData(BaseModel):
   Age: int
   Sex: str
   ChestPainType: str
   RestingBP: int
   Cholesterol: int
   FastingBS: int
   RestingECG: str
   MaxHR: int
   ExerciseAngina: str
   Oldpeak: float
   ST_Slope: str


@app.post("/predict")
def predict(data: InputData):
   input_data = pd.DataFrame([data.dict()])
  
   input_data['Sex'] = le_sex.transform(input_data['Sex'])
   input_data['ChestPainType'] = le_chestpaintype.transform(input_data['ChestPainType'])
   input_data['RestingECG'] = le_restingecg.transform(input_data['RestingECG'])
   input_data['ExerciseAngina'] = le_exerciseangina.transform(input_data['ExerciseAngina'])
   input_data['ST_Slope'] = le_stslope.transform(input_data['ST_Slope'])
   prediction = model.predict(input_data)


   return {"prediction": int(prediction[0])}
