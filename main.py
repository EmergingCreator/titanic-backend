from fastapi import FastAPI,HTTPException
from fastapi.middleware.cors import  CORSMiddleware
import joblib
from pydantic import BaseModel,Field
import numpy as np


app = FastAPI()

app.add_middleware(
     CORSMiddleware,
    allow_origins=["http://localhost:5173","https://titanic-frontend-zeta.vercel.app/"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class Data(BaseModel):
        pclass:int = Field(...,ge=1,le=3)
        age:int= Field(...,ge=1)
        fare:int = Field(...,ge=1)
        familySize:int = Field(...,ge=0)
        sex:int = Field(...,ge=0,le=1)
        embarked:int = Field(...,ge=0,le=2)


model = joblib.load("titanic_model.pkl")
@app.post("/predict")
def getPrediction(data:Data):
    try:
        ageGroup =0 
        if(data.age < 12):
            ageGroup= 0
        elif (data.age<18):
            ageGroup=  1
        elif (data.age<35):
            ageGroup=  2
        elif (data.age<60):
            ageGroup=  3
        else:
            ageGroup=  4


        isAlone=0
        if(data.familySize==1):
            isAlone=1 

        x=model.predict(np.array([[data.pclass,data.sex,data.age,data.fare,data.embarked,data.familySize,isAlone,ageGroup]]))[0]
        probability= model.predict_proba(np.array([[data.pclass,data.sex,data.age,data.fare,data.embarked,data.familySize,isAlone,ageGroup]]))[0]
        print(probability)
        return { "survived": int(x), "probability": probability.tolist() }

    except Exception as e:
         raise HTTPException(status_code=500,detail=f"Prediction Failed, Try Again! {e}" )
     