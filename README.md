# 🌍 TerraYield AI | Live Harvest Forecasting Dashboard

[![Live Demo](https://img.shields.io/badge/DEMO-Live_Deployment-brightgreen?style=for-the-badge&logo=vercel)](YOUR_DEPLOYMENT_LINK_HERE)

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Flask](https://img.shields.io/badge/Flask-3.0.3-lightgrey.svg)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3.2-orange.svg)
![Bootstrap](https://img.shields.io/badge/Bootstrap-5.3-purple.svg)

**TerraYield AI** is a professional-grade Precision Agriculture dashboard. It leverages a Machine Learning pipeline to forecast crop yields based on historical environmental data, while integrating live geospatial and weather APIs for real-time monitoring.

---

## 🔗 Live Deployment
> **Check out the live application here:** [**TerraYield AI Web Dashboard**](https://predicting-crop-yields.vercel.app/)  
*(Hosted on Vercel/Render/PythonAnywhere)*

---

## 📸 Project Gallery

### 1. Main Analytical Dashboard
![Dashboard Main View](https://github.com/kamilperwez/Predicting-Crop-Yields/blob/main/static/main.png)
*The primary interface featuring glassmorphism design, predictive inputs, and live weather telemetry.*

### 2. Predictive Insights & Comparisons
![Chart View](https://github.com/kamilperwez/Predicting-Crop-Yields/blob/main/static/chart.png)
*Comparative bar charts visualizing model output.*

### 3. Space-to-Earth Monitoring
![Satellite Widget](https://github.com/kamilperwez/Predicting-Crop-Yields/blob/main/static/geo.png)
*Real-time orbital tracking of the ISS to simulate satellite-based NDVI crop health monitoring.*

---

## ✨ Key Features

* **🧠 ML Predictive Engine:** Uses a Decision Tree Regressor to forecast yields (hg/ha) based on Rainfall, Pesticides, and Temperature.
* **📡 Live Geospatial Integration:** Fetches real-time capital city data and flag icons via **REST Countries API**.
* **🌤️ Dynamic Climatology:** Real-time weather conditions and wind speed sourced via **Open-Meteo API**.
* **🛰️ Satellite Telemetry:** Live tracking of the International Space Station (ISS) to demonstrate the data architecture of modern "Space-Ag" technology.
* **🛡️ Outlier Protection:** Built-in "Planetary Sanity Checks" ensure that user inputs remain within realistic agricultural boundaries.

## 🛠️ Installation & Local Setup

1.  **Clone the Repo:**
    ```bash
    git clone [https://github.com/YourUsername/TerraYield-AI.git](https://github.com/YourUsername/TerraYield-AI.git)
    cd TerraYield-AI
    ```

2.  **Install Requirements:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Launch App:**
    ```bash
    python app.py
    ```

## 📂 Data & Models
The project relies on a serialized `preprocessor.pkl` and `dtr.pkl` located in the `/models` directory, and a cleaned version of the FAO global agriculture dataset located in `/notebook_and_dataset`.

---
### ⚠️ Disclaimer
This tool uses historical data (1990-2013) to generate predictive forecasts. Results are estimates intended for educational and portfolio demonstration.

**Developed with ❤️ by [Your Name]**
