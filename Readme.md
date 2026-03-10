# House Price Prediction API

A deep learning model built with Keras to predict house prices, deployed using FastAPI.

---

## Project Structure

```
├── main.py
├── my_house_model.keras
├── scaler_weights.pkl
├── kc_house_data.csv
└── 01_Keras_Regression_task.ipynb
```

---

## How to Run

```bash
uvicorn main:app --reload
```

Then open: http://127.0.0.1:8000/docs

---

## Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Welcome message |
| POST | `/predict` | Predict house price |

---

## Example Request

```json
{
  "bedrooms": 3,
  "bathrooms": 2.25,
  "sqft_living": 1800,
  "sqft_lot": 7200,
  "floors": 2,
  "waterfront": 0,
  "view": 0,
  "condition": 3,
  "grade": 7,
  "sqft_above": 1800,
  "sqft_basement": 0,
  "yr_built": 1990,
  "yr_renovated": 0,
  "zipcode": 98178,
  "lat": 47.5112,
  "long": -122.257,
  "sqft_living15": 1340,
  "sqft_lot15": 5650,
  "month": 5,
  "year": 2015
}
```

## Example Response

```json
{
  "status": "success",
  "predicted_price": "$351,621.62"
}
```

---

## Dataset

King County House Sales dataset (kc_house_data.csv) — 21,613 records with 20 features.

## Model

- Framework: TensorFlow / Keras
- Type: Sequential Neural Network
- Loss: Mean Squared Error
- Scaler: MinMaxScaler