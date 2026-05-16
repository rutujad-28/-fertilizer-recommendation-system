# Fertilizer Recommendation System

A complete AI-powered Flask web application that recommends fertilizer based on soil nutrients, weather values, moisture, soil type, and crop type.

## Features

- Modern responsive Bootstrap 5 UI
- User authentication with login, registration, logout, and forgot-password reset
- Random Forest Classifier with Scikit-learn
- Crop recommendation model using soil nutrients, weather, temperature, and rainfall
- SQLite prediction history
- Weather autofill using OpenWeather API
- English and Marathi language toggle
- Voice input for numeric values
- Dashboard charts with Chart.js
- PDF recommendation report download
- Render-ready deployment files

## Project Structure

```text
fertilizer-recommendation-system/
├── static/
│   ├── css/
│   ├── js/
│   └── images/
├── templates/
├── model/
│   ├── fertilizer_dataset.csv
│   ├── train_model.py
│   └── model.pkl
├── database/
│   └── fertilizer.db
├── app.py
├── requirements.txt
├── Procfile
├── render.yaml
└── README.md
```

## Local Setup

1. Create and activate a virtual environment.

```bash
python -m venv .venv
.venv\Scripts\activate
```

2. Install dependencies.

```bash
pip install -r requirements.txt
```

3. Train the models.

```bash
python model/train_model.py
python model/train_crop_model.py
```

4. Optional: enable weather autofill.

```bash
set OPENWEATHER_API_KEY=your_api_key_here
```

5. Start the Flask app.

```bash
python app.py
```

Open `http://127.0.0.1:5000`.

## Machine Learning

The fertilizer model uses these input columns:

- Nitrogen
- Phosphorus
- Potassium
- Temperature
- Humidity
- Moisture
- Soil_Type
- Crop_Type

The target column is `Fertilizer_Name`. Categorical values are encoded with `OneHotEncoder`, and the final classifier is `RandomForestClassifier`.

The crop recommendation model uses:

- Nitrogen
- Phosphorus
- Potassium
- Temperature
- Humidity
- Rainfall

The crop target column is `Crop`, trained with a Random Forest classification pipeline.

## Render Deployment

This repository includes `render.yaml` and `Procfile`.

Set the following environment variables on Render:

- `SECRET_KEY`
- `OPENWEATHER_API_KEY` for weather autofill

The Render build command trains the model during deployment.

## Notes

This is an educational smart farming project. For real field usage, validate recommendations with local soil testing, crop stage, fertilizer labels, and regional agricultural guidance.
