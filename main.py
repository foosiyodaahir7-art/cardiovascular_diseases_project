from fastapi import FastAPI, Form, Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import numpy as np
import joblib
import os

app = FastAPI()

templates = Jinja2Templates(directory="web_interface")
rf_model = joblib.load("models/Models/random_forest_model.pkl")
lr_model = joblib.load("models/Models/logistic_model.pkl")

scaler = joblib.load("models/Models/cv_scaler.pkl")

@app.get("/")
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "results": None})

@app.post("/predict")
async def predict(
    request: Request,
    age: int = Form(...),
    gender: int = Form(...),
    height: int = Form(...),
    weight: float = Form(...),
    ap_hi: int = Form(...),
    ap_lo: int = Form(...),
    cholesterol: int = Form(...),
    glucose: int = Form(...),
    smoke: int = Form(...),
    physical_activity: int = Form(...)
):
    try:
        
        raw_features = np.array([[
            age * 365, gender, height, weight, ap_hi, ap_lo, 
            cholesterol, glucose, smoke, physical_activity
        ]])
        

        features_scaled = scaler.transform(raw_features)
        rf_pred = rf_model.predict(features_scaled)[0]
        lr_pred = lr_model.predict(features_scaled)[0]
        
  
        results = {
            "rf": "Healthy" if rf_pred == 0 else "High Risk",
            "rf_class": "healthy" if rf_pred == 0 else "risk",
            "rf_acc": "73.4%",
            "lr": "Healthy" if lr_pred == 0 else "High Risk",
            "lr_class": "healthy" if lr_pred == 0 else "risk",
            "lr_acc": "70.5%"
        }
        
        return templates.TemplateResponse("index.html", {"request": request, "results": results})
        
    except Exception as e:
        return {"Error": str(e)}