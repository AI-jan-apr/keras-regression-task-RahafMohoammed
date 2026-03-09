from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np
from tensorflow.keras.models import load_model

app = FastAPI()

class HouseFeatures(BaseModel):
    bedrooms: float
    bathrooms: float
    sqft_living: float
    sqft_lot: float
    floors: float
    waterfront: int
    view: int
    condition: int
    grade: int
    sqft_above: float
    sqft_basement: float
    yr_built: int
    yr_renovated: int
    zipcode: int
    lat: float
    long: float
    sqft_living15: float
    sqft_lot15: float
    month: int
    year: int

with open('scaler_weights.pkl', 'rb') as f:
    scaler = pickle.load(f)

model = load_model('my_house_model.keras')

@app.get("/")
def home():
    return {"message": "Welcome to the House Price Prediction API!"}

@app.post("/predict")
def predict_price(house: HouseFeatures):
    input_data = [[
        house.bedrooms, house.bathrooms, house.sqft_living, house.sqft_lot, 
        house.floors, house.waterfront, house.view, house.condition, 
        house.grade, house.sqft_above, house.sqft_basement, house.yr_built, 
        house.yr_renovated, house.zipcode, house.lat, house.long, 
        house.sqft_living15, house.sqft_lot15, house.month, house.year
    ]]
    
    scaled_data = scaler.transform(input_data)
    
    prediction = model.predict(scaled_data)
    
    return {
        "status": "success",
        "predicted_price": f"${float(prediction[0][0]):,.2f}"
    }
