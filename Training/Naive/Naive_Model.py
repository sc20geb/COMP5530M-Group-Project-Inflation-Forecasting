import numpy as np

class NaiveModel:
    def __init__(self):
        self.last_value = None
    
    def fit(self, series):
        """Store the last observed value from the training series."""
        self.last_value = series.iloc[-1]
        return self.last_value
    
    def predict(self, n_steps=1):
        """Predict the last observed value for the next n steps."""
        if self.last_value == None:
            raise ValueError("Model is not fitted yet. Call 'fit' with training data first.")
        return np.full(n_steps, self.last_value)