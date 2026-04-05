from flask import Flask, request, render_template
import numpy as np
import pandas as pd
import pickle
import os

app = Flask(__name__)

# --- Dynamic Path Resolution ---
# This ensures paths work on your local Windows and Vercel's Linux servers
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def load_pickle(rel_path):
    # rel_path will be something like 'models/dtr.pkl'
    full_path = os.path.join(BASE_DIR, rel_path)
    return pickle.load(open(full_path, 'rb'))

# 1. Load your ML files from the 'models' folder
dtr = load_pickle('models/dtr.pkl')
preprocessor = load_pickle('models/preprocessor.pkl')

# 2. Load unique values from the 'dataset' folder
# CHANGED: Added 'dataset' to the path
CSV_PATH = os.path.join(BASE_DIR, 'notebook_and_dataset', 'yield_df.csv')
df = pd.read_csv(CSV_PATH)
AREAS = sorted(df['Area'].unique().tolist())
ITEMS = sorted(df['Item'].unique().tolist())

@app.route('/')
def index():
    return render_template('index.html', areas=AREAS, items=ITEMS)

@app.route("/predict", methods=['POST'])
def predict():
    try:
        # Convert numeric inputs to correct types
        year = int(request.form['Year'])
        rain = float(request.form['average_rain_fall_mm_per_year'])
        pesticides = float(request.form['pesticides_tonnes'])
        temp = float(request.form['avg_temp'])
        area = request.form['Area']
        item = request.form['Item']

        # Prepare features for preprocessor
        features = np.array([[year, rain, pesticides, temp, area, item]], dtype=object)
        transformed_features = preprocessor.transform(features)
        
        # Get prediction
        prediction = dtr.predict(transformed_features)[0]

        return render_template('index.html', 
                               prediction=round(prediction, 2), 
                               areas=AREAS, 
                               items=ITEMS,
                               last_input=request.form)
                               
    except Exception as e:
        # This will show you the exact error on the webpage if it fails
        return render_template('index.html', 
                               error=f"Prediction Error: {str(e)}", 
                               areas=AREAS, 
                               items=ITEMS)

if __name__ == "__main__":
    app.run(debug=True)