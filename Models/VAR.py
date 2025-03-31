from statsmodels.tsa.api import VAR
import pandas as pd

class VARModel:
    def __init__(self, max_lag=1):
        self.max_lag = max_lag
        self.model_fit = None
        self.k_ar = None

    def fit(self, train_df: pd.DataFrame, maxlags: int = 1):
        model = VAR(train_df)
        self.model_fit = model.fit(maxlags=maxlags, ic='aic')
        self.k_ar = self.model_fit.k_ar
        return self.model_fit

    def forecast(self, input_data, steps: int):
        if self.model_fit is None:
            raise ValueError("Model not fitted. Call fit() first.")
        return self.model_fit.forecast(y=input_data, steps=steps)

    def summary(self):
        if self.model_fit is None:
            raise ValueError("Model not fitted. Call fit() first.")
        return self.model_fit.summary()
