import pickle

from .config import MODEL_PATH


class CO2EmissionPredictor:
    def __init__(self, model_path=MODEL_PATH):
        self.model_path = model_path
        self.model = self._load_model()

    def _load_model(self):
        with open(self.model_path, "rb") as model_file:
            return pickle.load(model_file)

    def predict(self, vehicle_input):
        return float(self.model.predict(vehicle_input.to_model_frame())[0])
