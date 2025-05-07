
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error


class ARIMAForecaster:
    def __init__(self, df: pd.DataFrame, target_col: str, order=(1, 1, 1)):
        self.df = df.sort_values("datetime")
        self.target_col = target_col
        self.order = order
        self.model = None
        self.fitted_model = None
    
    
    def run(self):
        self.df = self.df.dropna(subset=[self.target_col])  # Ensure no NaNs
        if len(self.df) < 30:
            raise ValueError("Too few non-null data points for forecasting")

        split_idx = int(len(self.df) * 0.8)
        train_df = self.df.iloc[:split_idx]
        test_df = self.df.iloc[split_idx:]

        if test_df.empty:
            raise ValueError("No data left for testing after split.")

        try:
            self.model = ARIMA(train_df[self.target_col], order=self.order)
            self.fitted_model = self.model.fit()
            forecast = self.fitted_model.forecast(steps=len(test_df))
        except Exception as e:
            print("ARIMA fitting failed:", e)
            raise

        if forecast.isna().all():
            print("⚠️ Forecast is all NaNs")
        
        metrics = self.evaluate(test_df[self.target_col], forecast)
        print(f"Forecasting Metrics: {metrics}")

        self.plot_forecast(train_df, test_df, forecast)

        result_df = pd.DataFrame({
            "forecast": forecast.values,  # ensure alignment
            "actual": test_df[self.target_col].values
        }, index=test_df.index)

        return self.fitted_model, result_df, metrics



    def evaluate(self, true_values, predicted_values):
        mse = mean_squared_error(true_values, predicted_values)
        rmse = float(np.sqrt(mse))
        mape = mean_absolute_percentage_error(true_values, predicted_values)
        return {"RMSE": rmse, "MAPE": mape}

    def plot_forecast(self, train_df, test_df, forecast):
        plt.figure(figsize=(10, 5))
        plt.plot(train_df["datetime"], train_df[self.target_col], label='Train')
        plt.plot(test_df["datetime"], test_df[self.target_col], label='Test')
        plt.plot(test_df["datetime"], forecast, label='Forecast')
        plt.legend()
        plt.title("ARIMA Forecast")
        plt.xlabel("Datetime")
        plt.ylabel(self.target_col)
        plt.tight_layout()
        plt.show()


    """ def save_model(self, filepath):
        joblib.dump(self.fitted_model, filepath)
        print(f"ARIMA model saved to {filepath}") """




""" 
class ProphetForecaster:
    def __init__(self):
        # Enable uncertainty intervals
        self.model = Prophet(uncertainty_samples=1000)

    def fit(self, df):
        
        #Train Prophet model.
        # Expects df with columns 'ds' (datetime) and 'y' (target).
        
        self.model.fit(df)

    def forecast(self, future_df):
        
        # Generate forecast on future data (must contain 'ds' column).
        
        return self.model.predict(future_df)

    def evaluate(self, true_values, predicted_values):
        
        #Evaluate forecast using RMSE and MAPE.
        
        mse = mean_squared_error(true_values, predicted_values)
        rmse = np.sqrt(mse)
        mape = mean_absolute_percentage_error(true_values, predicted_values)
        return {"RMSE": rmse, "MAPE": mape}

        
    def plot_forecast(self, forecast):
       
        # Plot forecast with uncertainty intervals if available.
       
        if 'yhat_lower' not in forecast.columns or 'yhat_upper' not in forecast.columns:
            print("Warning: 'yhat_lower' and 'yhat_upper' not found. Plotting without uncertainty.")
            self.model.plot(forecast, uncertainty=False)
        else:
            self.model.plot(forecast, uncertainty=True)
        plt.title("Prophet Forecast")
        plt.xlabel("Date")
        plt.ylabel("Forecast")
        plt.show()
 """