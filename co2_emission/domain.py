FUEL_TYPE_LABELS = {
    "X": "Regular Gasoline",
    "Z": "Premium Gasoline",
    "D": "Diesel",
    "E": "Ethanol (E85)",
}


def classify_emission(co2_value):
    percentage = min(100, int((co2_value / 400) * 100))

    if co2_value < 170:
        return "Low", "low-emission", percentage
    if co2_value <= 280:
        return "Moderate", "moderate-emission", percentage
    return "High", "high-emission", percentage


def format_vehicle_class(vehicle_class):
    return vehicle_class.title().replace(" - ", " ")
