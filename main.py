from flask import Flask, render_template, request
from co2_emission.config import STATIC_DIR, TEMPLATE_DIR
from co2_emission.domain import classify_emission
from co2_emission.parser import VehicleInput
from co2_emission.predictor import CO2EmissionPredictor


app = Flask(__name__, template_folder=str(TEMPLATE_DIR), static_folder=str(STATIC_DIR))
predictor = CO2EmissionPredictor()


@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        try:
            vehicle_input = VehicleInput.from_form(request.form)
            predicted_co2 = predictor.predict(vehicle_input)
            level, level_class, percentage = classify_emission(predicted_co2)

            return render_template(
                'result.html',
                EMI=round(predicted_co2),
                level=level,
                level_class=level_class,
                percentage=percentage,
                features=vehicle_input.to_summary(),
            )

        except Exception as e:
            print(f"[ERROR] Prediction failed: {e}")
            return render_template('index.html', error=f"Something went wrong: {e}")

    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
