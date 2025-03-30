import pandas as pd
from statsmodels.tsa.api import VAR

class VARModel:
    def __init__(self, max_lag=5):
        """
        Initialize a VAR model with max_lag.
        """
        self.max_lag = max_lag
        self.model_fit = None
        self.k_ar = None

    def fit(self, train_df: pd.DataFrame):
        """
        Fit the VAR model on the training data.

        Parameters:
        -----------
        train_df : pd.DataFrame
            Time-series data to fit the model on.
        """
        model = VAR(train_df)
        self.model_fit = model.fit(maxlags=self.max_lag, ic='aic')
        self.k_ar = self.model_fit.k_ar
        return self.model_fit

    def forecast(self, input_data, steps: int):
        """
        Forecast future values using the fitted model.

        Parameters:
        -----------
        input_data : np.array
            The last k_ar values of training data to start forecasting.
        steps : int
            Number of steps to forecast into the future.

        Returns:
        --------
        forecast : np.array
        """
        if self.model_fit is None:
            raise ValueError("Model not fitted. Call fit() first.")
        return self.model_fit.forecast(y=input_data, steps=steps)

    def summary(self):
        """
        Return the model summary.
        """
        if self.model_fit is None:
            raise ValueError("Model not fitted. Call fit() first.")
        return self.model_fit.summary()
