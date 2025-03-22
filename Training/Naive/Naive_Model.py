import numpy as np

class NaiveModel:
    def __init__(self):
        self.last_value = None

    def predict(self, series, n_steps=1):
        """Predict the last observed value for the next n steps."""
        self.last_value = series.iloc[-1]
        return np.full(n_steps, self.last_value)