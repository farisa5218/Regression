from flask import Flask, render_template, request
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load model pipeline
with open('model.pkl', 'rb') as f:
    regr = pickle.load(f)

# Fuel Type Mapping for readable UI display
fuel_type_mapping = {
    'X': 'Regular Gasoline',
    'Z': 'Premium Gasoline',
    'D': 'Diesel',
    'E': 'Ethanol (E85)'
}

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        myDict = request.form
        try:
            engine = float(myDict.get('Engine', 2.0))
            cylinders = int(myDict.get('Cylinders', 4))
            fuel_type = myDict.get('Fuel_Type', 'X')
            vehicle_class = myDict.get('Vehicle_Class', 'MID-SIZE')
            transmission = myDict.get('Transmission', 'AS6')
            fc_city = float(myDict.get('Fuel_Cons_City', 10.0))
            fc_hwy = float(myDict.get('Fuel_Cons_Hwy', 8.0))
            
            # Construct DataFrame with exact column names used in model training
            input_data = pd.DataFrame({
                'ENGINESIZE': [engine],
                'CYLINDERS': [cylinders],
                'FUELCONSUMPTION_CITY': [fc_city],
                'FUELCONSUMPTION_HWY': [fc_hwy],
                'FUELTYPE': [fuel_type],
                'VEHICLECLASS': [vehicle_class],
                'TRANSMISSION': [transmission]
            })
            
            # Run prediction through pipeline (handles scaling and encoding automatically)
            predicted_co2 = regr.predict(input_data)[0]
            
            # Format inputs for summary display
            features_summary = {
                'Engine Size': f"{engine} L",
                'Cylinders': f"{cylinders}",
                'Fuel Type': fuel_type_mapping.get(fuel_type, fuel_type),
                'Vehicle Class': vehicle_class.title().replace(' - ', ' '),
                'Transmission': transmission,
                'Fuel Consumption (City)': f"{fc_city} L/100km",
                'Fuel Consumption (Highway)': f"{fc_hwy} L/100km"
            }
            
            # Classify CO2 emissions level
            if predicted_co2 < 170:
                level = "Low"
                level_class = "low-emission"
                percentage = min(100, int((predicted_co2 / 400) * 100))
            elif predicted_co2 <= 280:
                level = "Moderate"
                level_class = "moderate-emission"
                percentage = min(100, int((predicted_co2 / 400) * 100))
            else:
                level = "High"
                level_class = "high-emission"
                percentage = min(100, int((predicted_co2 / 400) * 100))
                
            return render_template(
                'result.html', 
                EMI=round(predicted_co2), 
                level=level, 
                level_class=level_class, 
                percentage=percentage,
                features=features_summary
            )
            
        except Exception as e:
            print("Prediction Error:", e)
            return render_template('index.html', error=f"Error processing input: {str(e)}")
            
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
