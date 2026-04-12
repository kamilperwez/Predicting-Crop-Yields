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
CSV_PATH = os.path.join(BASE_DIR, 'notebook_and_dataset', 'yield_df.csv')
try:
    df = pd.read_csv(CSV_PATH)
    AREAS = sorted(df['Area'].unique().tolist())
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

# --- REUSABLE PREDICTION ENGINE ---
def get_predictions(area, selected_items, rain, pest, temp):
    current_year = 2026
    results = []
    for item in selected_items[:5]: 
        features = np.array([[current_year, rain, pest, temp, area, item]], dtype=object)
        transformed = preprocessor.transform(features)
        pred = dtr.predict(transformed)[0]
        final_pred = max(0, round(pred, 2))
        results.append({'label': item, 'value': final_pred})
    return results

# ==========================================
# 4. FLASK ROUTES
# ==========================================
@app.route('/')
def index():
    # --- DEFAULT INDIA SHOWCASE ---
    default_area = "India"
    # Ensure these match crops actually available in India in your dataset
    default_items = ["Rice, paddy", "Wheat"] 
    default_rain = 1050.0
    default_pest = 45000.0
    default_temp = 26.5
    
    try:
        # Generate prediction immediately so the dashboard isn't empty
        results = get_predictions(default_area, default_items, default_rain, default_pest, default_temp)
        
        user_data = {
            "Target Region": default_area,
            "Crops Evaluated": ", ".join(default_items),
            "Rainfall": f"{default_rain} mm/yr",
            "Pesticides": f"{default_pest} Tonnes",
            "Avg Temp": f"{default_temp} °C"
        }
    except Exception as e:
        print(f"Error generating defaults: {e}")
        results = []
        user_data = {}

    # Pass {"Area": "India"} to form_data so ONLY the country dropdown is pre-selected, 
    # but the numbers remain empty (showing placeholders)
    initial_form_state = {"Area": default_area}

    return render_template('index.html', 
                           results=results, 
                           areas=AREAS, 
                           area_crop_map=AREA_CROP_MAP, 
                           capitals=CAPITALS, 
                           trends=FARMING_TRENDS,
                           last_area=default_area, 
                           trend=FARMING_TRENDS.get(default_area, DEFAULT_TREND),
                           user_data=user_data,
                           form_data=initial_form_state)

@app.route("/predict", methods=['POST'])
def predict():
    try:
        # Validation
        required_fields = ['average_rain_fall_mm_per_year', 'pesticides_tonnes', 'avg_temp', 'Area']
        if any(not request.form.get(field) or request.form.get(field).strip() == "" for field in required_fields):
            return render_template('index.html', error="⚠️ Validation Error: All form fields are mandatory.", areas=AREAS, area_crop_map=AREA_CROP_MAP, capitals=CAPITALS, trends=FARMING_TRENDS, last_area="India", form_data=None)

        selected_items = request.form.getlist('Item')
        if not selected_items:
            return render_template('index.html', error="⚠️ Validation Error: Please select at least one crop to analyze.", areas=AREAS, area_crop_map=AREA_CROP_MAP, capitals=CAPITALS, trends=FARMING_TRENDS, last_area="India", form_data=None)

        # Parse Inputs
        rain = float(request.form['average_rain_fall_mm_per_year'])
        pest = float(request.form['pesticides_tonnes'])
        temp = float(request.form['avg_temp'])
        area = request.form['Area']

        # Integrity Checks
        if not (0 <= rain <= 15000) or not (0 <= pest <= 500000) or not (-20 <= temp <= 60):
            return render_template('index.html', error="🛑 Data Integrity Error: Inputs fall outside realistic bounds.", areas=AREAS, area_crop_map=AREA_CROP_MAP, capitals=CAPITALS, trends=FARMING_TRENDS, last_area=area, form_data=None)

        # Generate Predictions
        results = get_predictions(area, selected_items, rain, pest, temp)
        
        # UI Data
        user_data = {
            "Target Region": area,
            "Crops Evaluated": ", ".join(selected_items),
            "Rainfall": f"{rain} mm/yr",
            "Pesticides": f"{pest} Tonnes",
            "Avg Temp": f"{temp} °C"
        }
        
        # Save Form Data to keep inputs filled after submission
        form_data = {
            "Area": area,
            "Item": selected_items,
            "average_rain_fall_mm_per_year": rain,
            "pesticides_tonnes": pest,
            "avg_temp": temp
        }

        return render_template('index.html', results=results, areas=AREAS, area_crop_map=AREA_CROP_MAP, 
                               capitals=CAPITALS, trends=FARMING_TRENDS, last_area=area, 
                               trend=FARMING_TRENDS.get(area, DEFAULT_TREND),
                               user_data=user_data, form_data=form_data)
                               
    except Exception as e:
        return render_template('index.html', error=f"System Error: {str(e)}", areas=AREAS, area_crop_map=AREA_CROP_MAP, capitals=CAPITALS, trends=FARMING_TRENDS, last_area="India", form_data=None)

if __name__ == "__main__":
    app.run(debug=True)