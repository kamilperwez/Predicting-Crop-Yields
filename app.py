from flask import Flask, request, render_template
import numpy as np
import pandas as pd
import pickle
import os

app = Flask(__name__)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ==========================================
# 1. SAFELY LOAD MACHINE LEARNING MODELS
# ==========================================
try:
    dtr = pickle.load(open(os.path.join(BASE_DIR, 'models', 'dtr.pkl'), 'rb'))
    preprocessor = pickle.load(open(os.path.join(BASE_DIR, 'models', 'preprocessor.pkl'), 'rb'))
except Exception as e:
    print(f"CRITICAL ERROR: Could not load models. Please ensure 'models/dtr.pkl' exists. {e}")

# ==========================================
# 2. SAFELY LOAD THE DATASET
# ==========================================
# Note: Path is explicitly set to 'notebook_and_dataset'
CSV_PATH = os.path.join(BASE_DIR, 'notebook_and_dataset', 'yield_df.csv')
try:
    df = pd.read_csv(CSV_PATH)
    AREAS = sorted(df['Area'].unique().tolist())
    # Create a dynamic mapping of which crops belong to which country
    AREA_CROP_MAP = {area: sorted(df[df['Area'] == area]['Item'].unique().tolist()) for area in AREAS}
except Exception as e:
    print(f"CRITICAL ERROR: Could not load CSV. Ensure 'notebook_and_dataset/yield_df.csv' exists. {e}")
    AREAS, AREA_CROP_MAP = [], {}

# ==========================================
# 3. KNOWLEDGE BASES (Capitals & Trends)
# ==========================================
CAPITALS = {
    "India": "New Delhi", "Brazil": "Brasilia", "USA": "Washington D.C.",
    "Albania": "Tirana", "Algeria": "Algiers", "Australia": "Canberra",
    "Pakistan": "Islamabad", "France": "Paris", "Spain": "Madrid",
    "Egypt": "Cairo", "Canada": "Ottawa", "Germany": "Berlin",
    "China": "Beijing", "Japan": "Tokyo", "United Kingdom": "London",
    "South Africa": "Pretoria", "Argentina": "Buenos Aires", "Mexico": "Mexico City",
    "Italy": "Rome", "Nigeria": "Abuja"
}

FARMING_TRENDS = {
    "India": "Scaling up solar-powered micro-irrigation and precision agriculture to combat erratic monsoons.",
    "USA": "Heavy investments in AI-driven autonomous tractors and regenerative soil practices.",
    "Brazil": "Expanding sustainable soy production and Integrated Crop-Livestock-Forestry (ICLF) systems.",
    "Australia": "Pioneering drought-resistant genetically modified crops and smart-water management IoT.",
    "France": "Transitioning rapidly towards agroecology and strict pesticide reduction under the Ecophyto plan.",
    "China": "Implementing drone-based crop monitoring and vertical farming in urban-adjacent mega-greenhouses.",
    "Japan": "Utilizing robotic harvesters and AI-controlled indoor farming to offset an aging agricultural workforce."
}
DEFAULT_TREND = "Integrating advanced satellite monitoring and regenerative soil techniques to boost organic yield thresholds."

# ==========================================
# 4. FLASK ROUTES
# ==========================================
@app.route('/')
def index():
    # India is set as the default landing page showcase
    return render_template('index.html', 
                           areas=AREAS, 
                           area_crop_map=AREA_CROP_MAP, 
                           capitals=CAPITALS, 
                           trends=FARMING_TRENDS,
                           last_area="India", 
                           trend=FARMING_TRENDS.get("India", DEFAULT_TREND))

@app.route("/predict", methods=['POST'])
def predict():
    try:
        # --- 1. STRICT BACKEND VALIDATION ---
        # Stop execution if a user bypassed the HTML required tags
        required_fields = ['average_rain_fall_mm_per_year', 'pesticides_tonnes', 'avg_temp', 'Area']
        if any(not request.form.get(field) or request.form.get(field).strip() == "" for field in required_fields):
            return render_template('index.html', error="⚠️ Validation Error: All form fields are mandatory.", 
                                   areas=AREAS, area_crop_map=AREA_CROP_MAP, capitals=CAPITALS, trends=FARMING_TRENDS, last_area="India")

        selected_items = request.form.getlist('Item')
        if not selected_items:
            return render_template('index.html', error="⚠️ Validation Error: Please select at least one crop to analyze.", 
                                   areas=AREAS, area_crop_map=AREA_CROP_MAP, capitals=CAPITALS, trends=FARMING_TRENDS, last_area="India")

        # --- 2. PARSE INPUTS ---
        current_year = 2026 # Fixed year for current/future predictions
        rain = float(request.form['average_rain_fall_mm_per_year'])
        pest = float(request.form['pesticides_tonnes'])
        temp = float(request.form['avg_temp'])
        area = request.form['Area']

        # --- 3. PLANETARY SANITY CHECKS (Outlier Prevention) ---
        # This prevents the ML Model from "hallucinating" if fed physically impossible data
        if not (0 <= rain <= 15000):
            return render_template('index.html', error="🛑 Data Integrity Error: Rainfall must be between 0 and 15,000 mm/yr to match planetary norms.", 
                                   areas=AREAS, area_crop_map=AREA_CROP_MAP, capitals=CAPITALS, trends=FARMING_TRENDS, last_area=area)
        
        if not (0 <= pest <= 500000):
            return render_template('index.html', error="🛑 Data Integrity Error: Pesticide usage exceeds realistic national bounds (Max: 500,000 Tonnes).", 
                                   areas=AREAS, area_crop_map=AREA_CROP_MAP, capitals=CAPITALS, trends=FARMING_TRENDS, last_area=area)
        
        if not (-20 <= temp <= 60):
            return render_template('index.html', error="🛑 Data Integrity Error: Temperature must be between -20°C and 60°C to simulate viable agriculture.", 
                                   areas=AREAS, area_crop_map=AREA_CROP_MAP, capitals=CAPITALS, trends=FARMING_TRENDS, last_area=area)

        # --- 4. PREDICTION LOGIC ---
        country_trend = FARMING_TRENDS.get(area, DEFAULT_TREND)
        results = []
        
        for item in selected_items[:5]: 
            # Feature mapping must be EXACTLY: [Year, Rain, Pest, Temp, Area, Item]
            features = np.array([[current_year, rain, pest, temp, area, item]], dtype=object)
            transformed = preprocessor.transform(features)
            pred = dtr.predict(transformed)[0]
            
            # Ensure no negative predictions are passed to the frontend
            final_pred = max(0, round(pred, 2))
            results.append({'label': item, 'value': final_pred})

        # Return successful predictions to the UI
        return render_template('index.html', results=results, areas=AREAS, area_crop_map=AREA_CROP_MAP, 
                               capitals=CAPITALS, trends=FARMING_TRENDS, last_area=area, trend=country_trend)
                               
    except ValueError:
        return render_template('index.html', error="⚠️ Data Error: Please enter valid numbers for Rainfall, Pesticides, and Temp.", 
                               areas=AREAS, area_crop_map=AREA_CROP_MAP, capitals=CAPITALS, trends=FARMING_TRENDS, last_area="India")
    except Exception as e:
        return render_template('index.html', error=f"System Error: {str(e)}", 
                               areas=AREAS, area_crop_map=AREA_CROP_MAP, capitals=CAPITALS, trends=FARMING_TRENDS, last_area="India")

if __name__ == "__main__":
    app.run(debug=True)