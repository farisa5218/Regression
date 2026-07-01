# =============================================================================
# main.py — Flask Web Application for CO2 Emission Prediction
# =============================================================================
# How it works:
#   1. User fills a form on the website with their car details.
#   2. Flask receives the form data (POST request).
#   3. Data is fed into the trained ML model (model.pkl).
#   4. The predicted CO2 emission is shown on the result page.
# =============================================================================

from flask import Flask, render_template, request
import pickle
import pandas as pd

# ── App Setup ─────────────────────────────────────────────────────────────────
app = Flask(__name__)

# Load the pre-trained ML model pipeline from disk
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# Maps short fuel-type codes to human-readable names for display
FUEL_TYPE_LABELS = {
    'X': 'Regular Gasoline',
    'Z': 'Premium Gasoline',
    'D': 'Diesel',
    'E': 'Ethanol (E85)',
}

# ── Helper: Classify CO2 Level ────────────────────────────────────────────────
def classify_emission(co2_value):
    """
    Returns emission level label, CSS class, and gauge percentage
    based on the predicted CO2 value (in g/km).

    Thresholds:
        < 170  g/km  → Low      (green)
        170–280 g/km → Moderate (yellow)
        > 280  g/km  → High     (red)
    """
    percentage = min(100, int((co2_value / 400) * 100))

    if co2_value < 170:
        return "Low", "low-emission", percentage
    elif co2_value <= 280:
        return "Moderate", "moderate-emission", percentage
    else:
        return "High", "high-emission", percentage


# ── Main Route ────────────────────────────────────────────────────────────────
@app.route("/", methods=["GET", "POST"])
def home():
    """
    GET  → Shows the input form (index.html).
    POST → Reads form data, runs prediction, shows result (result.html).
    """
    if request.method == "POST":
        try:
            # ── Step 1: Read user inputs from the form ──────────────────────
            engine      = float(request.form.get('Engine', 2.0))
            cylinders   = int(request.form.get('Cylinders', 4))
            fuel_type   = request.form.get('Fuel_Type', 'X')
            vehicle_cls = request.form.get('Vehicle_Class', 'MID-SIZE')
            transmission = request.form.get('Transmission', 'AS6')
            fc_city     = float(request.form.get('Fuel_Cons_City', 10.0))
            fc_hwy      = float(request.form.get('Fuel_Cons_Hwy', 8.0))

            # ── Step 2: Build a DataFrame matching the model's training format ─
            input_df = pd.DataFrame({
                'ENGINESIZE':           [engine],
                'CYLINDERS':            [cylinders],
                'FUELCONSUMPTION_CITY': [fc_city],
                'FUELCONSUMPTION_HWY':  [fc_hwy],
                'FUELTYPE':             [fuel_type],
                'VEHICLECLASS':         [vehicle_cls],
                'TRANSMISSION':         [transmission],
            })

            # ── Step 3: Predict (pipeline handles scaling & encoding internally) ─
            predicted_co2 = model.predict(input_df)[0]

            # ── Step 4: Classify the result ─────────────────────────────────
            level, level_class, percentage = classify_emission(predicted_co2)

            # ── Step 5: Build a summary dict for the result page ────────────
            summary = {
                'Engine Size':                  f"{engine} L",
                'Cylinders':                    f"{cylinders}",
                'Fuel Type':                    FUEL_TYPE_LABELS.get(fuel_type, fuel_type),
                'Vehicle Class':                vehicle_cls.title(),
                'Transmission':                 transmission,
                'Fuel Consumption (City)':      f"{fc_city} L/100km",
                'Fuel Consumption (Highway)':   f"{fc_hwy} L/100km",
            }

            return render_template(
                'result.html',
                EMI=round(predicted_co2),
                level=level,
                level_class=level_class,
                percentage=percentage,
                features=summary,
            )

        except Exception as e:
            # Show an error message on the form page if something goes wrong
            print(f"[ERROR] Prediction failed: {e}")
            return render_template('index.html', error=f"Something went wrong: {e}")

    # Default: render the input form
    return render_template('index.html')


# ── Run the App ───────────────────────────────────────────────────────────────
if __name__ == '__main__':
    app.run(debug=True)
