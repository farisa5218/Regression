from dataclasses import dataclass

import pandas as pd

from .domain import FUEL_TYPE_LABELS, format_vehicle_class


@dataclass(frozen=True)
class VehicleInput:
    engine: float
    cylinders: int
    fuel_type: str
    vehicle_class: str
    transmission: str
    fuel_consumption_city: float
    fuel_consumption_highway: float

    @classmethod
    def from_form(cls, form):
        return cls(
            engine=float(form.get("Engine", 2.0)),
            cylinders=int(form.get("Cylinders", 4)),
            fuel_type=form.get("Fuel_Type", "X"),
            vehicle_class=form.get("Vehicle_Class", "MID-SIZE"),
            transmission=form.get("Transmission", "AS6"),
            fuel_consumption_city=float(form.get("Fuel_Cons_City", 10.0)),
            fuel_consumption_highway=float(form.get("Fuel_Cons_Hwy", 8.0)),
        )

    def to_model_frame(self):
        return pd.DataFrame({
            "ENGINESIZE": [self.engine],
            "CYLINDERS": [self.cylinders],
            "FUELCONSUMPTION_CITY": [self.fuel_consumption_city],
            "FUELCONSUMPTION_HWY": [self.fuel_consumption_highway],
            "FUELTYPE": [self.fuel_type],
            "VEHICLECLASS": [self.vehicle_class],
            "TRANSMISSION": [self.transmission],
        })

    def to_summary(self):
        return {
            "Engine Size": f"{self.engine} L",
            "Cylinders": f"{self.cylinders}",
            "Fuel Type": FUEL_TYPE_LABELS.get(self.fuel_type, self.fuel_type),
            "Vehicle Class": format_vehicle_class(self.vehicle_class),
            "Transmission": self.transmission,
            "Fuel Consumption (City)": f"{self.fuel_consumption_city} L/100km",
            "Fuel Consumption (Highway)": f"{self.fuel_consumption_highway} L/100km",
        }
