import pandas as pd
import os
import joblib
from typing import Dict
from modeling import ARIMAForecaster
from visualization import GeohashForecastVisualizer
from sklearn.metrics import mean_absolute_percentage_error

class ARIMAForecastingPipeline:
    def __init__(self, df: pd.DataFrame, target_col: str, city_name: str, geohash_col: str = 'geohash'):
        self.df = df.copy()
        self.target_col = target_col
        self.city_name = city_name
        self.geohash_col = geohash_col
        self.models_dir = f"../models/{city_name}"
        self.forecasts = {}
        self.geohash_forecasts = []

        os.makedirs(self.models_dir, exist_ok=True)

    def run(self):
        all_metrics = []

        if self.geohash_col not in self.df.columns:
            print("Geohash column not found. Running city-level forecast only.")
            self._run_city_forecast(all_metrics)
        else:
            print("Running both city-level and geohash-level forecasts.")
            self._run_city_forecast(all_metrics)
            self._run_geohash_forecasts(all_metrics)
            

        self._report_average_metrics(all_metrics)

        self.diagnose_forecasts(self.forecasts, city_name=self.city_name)

        visualizer = GeohashForecastVisualizer(self.forecasts, self.city_name)
        visualizer.plot_forecast_heatmap()
        visualizer.plot_forecast_time_series()


    def _run_city_forecast(self, metrics_list):
        print(f"\n--- Running City-Level Forecast for {self.city_name} ---")
        forecaster = ARIMAForecaster(self.df, self.target_col)
        
        model, forecast, metrics = forecaster.run()
        metrics_list.append(metrics)

        model_path = os.path.join(self.models_dir, f"{self.city_name}_arima_model.pkl")
        joblib.dump(model, model_path)

        self.forecasts['city'] = forecast
        print(f"Saved city-level model to {model_path}\n")

    def _run_geohash_forecasts(self, metrics_list):
        grouped = self.df.groupby(self.geohash_col)
        print(f"🧪 Type of metrics_list: {type(metrics_list)}")

        for geohash, group in grouped:
            print(f"\n🔲 Running ARIMA forecast for {self.city_name} - Geohash: {geohash}")
            group_df = group.sort_values("datetime")
            if len(group_df) < 30:
                print(f"⚠️ Skipping geohash {geohash} due to insufficient data ({len(group_df)} rows)")
                continue
            forecaster = ARIMAForecaster(group_df, self.target_col)

            try:
                model, forecast, metrics = forecaster.run()
                self.geohash_forecasts.append({
                "type": "geohash",
                "geohash": geohash,
                "forecast": forecast,
                "metrics": metrics
            })

                metrics_list.append(metrics)
                print(f"✅ Completed geohash {geohash}. RMSE: {metrics['RMSE']:.2f}, MAPE: {metrics['MAPE']:.2f}")
            except Exception as e:
                print(f"❌ Error processing geohash {geohash}: {e}")

    @staticmethod
    def _report_average_metrics(metrics_list):
        if not metrics_list:
            print("No metrics to report.")
            return

        avg_rmse = sum(m["RMSE"] for m in metrics_list) / len(metrics_list)
        avg_mape = sum(m["MAPE"] for m in metrics_list) / len(metrics_list)

        print(f"\n📊 Average RMSE: {avg_rmse:.3f}")
        print(f"📊 Average MAPE: {avg_mape:.3f}")


    @staticmethod
    def diagnose_forecasts(forecasts: Dict[str, pd.DataFrame], city_name: str = "") -> None:
        print(f"\n🔍 Diagnosing forecast entries for {city_name}...\n")
        for geohash, df in forecasts.items():
            if not isinstance(df, pd.DataFrame):
                print(f"[{geohash}] ❌ Not a DataFrame")
                continue
            if 'forecast' not in df.columns:
                print(f"[{geohash}] ❌ Missing 'forecast' column")
                continue
            if df['forecast'].dropna().empty:
                print(f"[{geohash}] ⚠️  'forecast' column is all NaNs")
            else:
                print(f"[{geohash}] ✅ OK – {df['forecast'].dropna().shape[0]} forecast values")

