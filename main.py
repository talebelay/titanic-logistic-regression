from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse
import joblib
import numpy as np

app = FastAPI()
model = joblib.load("logistic_model.joblib") # Change to decision_tree_model.joblib for the 2nd repo

@app.get("/", response_class=HTMLResponse)
async def home():
    with open("index.html") as f:
        return f.read()

@app.post("/predict")
async def predict(pclass: int = Form(...), sex: int = Form(...), age: float = Form(...), 
                  sibsp: int = Form(...), parch: int = Form(...), fare: float = Form(...), 
                  embarked: int = Form(...)):
    features = np.array([[pclass, sex, age, sibsp, parch, fare, embarked]])
    prediction = model.predict(features)[0]
    result = "Survived! 🎉" if prediction == 1 else "Did not survive 🛑"
    return {"prediction": result}