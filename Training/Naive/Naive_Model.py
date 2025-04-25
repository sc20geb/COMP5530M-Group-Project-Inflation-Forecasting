import numpy as np

class NaiveModel:
    def __init__(self):
        self.last_value = None

    def predict(self, lastValue, n_steps=1):
        """Predict the last observed value for the next n steps."""
        self.last_value = lastValue
        return np.full(n_steps, self.last_value)