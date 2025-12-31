from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse
import joblib
import numpy as np

app = FastAPI()
# Ensure this matches your file name in the repo
model = joblib.load("logistic_model.joblib")

@app.get("/", response_class=HTMLResponse)
async def home():
    with open("index.html") as f:
        return f.read()

@app.post("/predict")
async def predict(pclass: int = Form(...), sex: int = Form(...), age: float = Form(...), 
                  sibsp: int = Form(...), parch: int = Form(...), fare: float = Form(...), 
                  embarked: int = Form(...)):
    
    features = np.array([[pclass, sex, age, sibsp, parch, fare, embarked]])
    
    # 0 or 1 prediction
    prediction = model.predict(features)[0]
    
    # Calculate probability/confidence
    probabilities = model.predict_proba(features)[0]
    # If survived (1), take the second value; if not (0), take the first.
    confidence = probabilities[1] if prediction == 1 else probabilities[0]
    
    result = "Survived" if prediction == 1 else "Did Not Survive"
    
    return {
        "prediction": result,
        "confidence": f"{round(confidence * 100, 2)}%"
    }
