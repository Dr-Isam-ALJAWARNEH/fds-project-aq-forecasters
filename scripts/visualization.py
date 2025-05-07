import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict

class GeohashForecastVisualizer:
    def __init__(self, forecasts: Dict[str, pd.DataFrame], city_name: str):
        self.forecasts = forecasts
        self.city_name = city_name

    def plot_forecast_for_geohash(self, geohash: str) -> None:
        if geohash not in self.forecasts:
            print(f"No forecast found for geohash: {geohash}")
            return

        df = self.forecasts[geohash]
        plt.figure(figsize=(14, 6))
        plt.plot(df.index, df['actual'], label='Actual')
        plt.plot(df.index, df['forecast'], label='Forecast', linestyle='--')
        plt.title(f"Forecast vs Actual for {geohash} in {self.city_name}")
        plt.xlabel("Time")
        plt.ylabel("PM2.5")
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.show()

    def plot_all_geohash_forecasts(self, ncols: int = 3) -> None:
        n = len(self.forecasts)
        nrows = (n + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
        axes = axes.flatten()

        for idx, (geohash, df) in enumerate(self.forecasts.items()):
            ax = axes[idx]
            ax.plot(df.index, df['actual'], label='Actual')
            ax.plot(df.index, df['forecast'], label='Forecast', linestyle='--')
            ax.set_title(f"{geohash}")
            ax.tick_params(axis='x', rotation=45)

        for i in range(idx + 1, len(axes)):
            axes[i].axis('off')

        plt.suptitle(f"Forecast vs Actual for All Geohashes in {self.city_name}", fontsize=16)
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper right')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

    def plot_forecast_errors_distribution(self) -> None:
        errors = {
            geo: (df['actual'] - df['forecast']).dropna()
            for geo, df in self.forecasts.items()
        }
        error_df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in errors.items()]))

        plt.figure(figsize=(14, 6))
        sns.boxplot(data=error_df, orient='h')
        plt.title(f"Forecast Error Distribution per Geohash in {self.city_name}")
        plt.xlabel("Error (Actual - Forecast)")
        plt.grid()
        plt.show()

    def plot_forecast_heatmap(self, n_geohashes: int = 20) -> None:
        # Get the last forecast value per geohash
        last_forecast = {
            geo: df['forecast'].dropna().iloc[-1]
            for geo, df in self.forecasts.items()
            if geo != 'city' and 'forecast' in df and not df['forecast'].dropna().empty
        }

        if not last_forecast:
            print("No forecast data available for heatmap.")
            return

        heatmap_df = pd.DataFrame.from_dict(last_forecast, orient='index', columns=['LastForecast'])

        # Take top `n_geohashes` by forecasted value
        heatmap_df = heatmap_df.sort_values('LastForecast', ascending=False).head(n_geohashes)

        plt.figure(figsize=(10, 0.5 * len(heatmap_df)))  # Adjust height dynamically
        sns.heatmap(
            heatmap_df,
            annot=True,
            cmap="YlGnBu",
            fmt=".1f",
            cbar_kws={"label": "PM2.5"}
        )
        plt.title(f"{self.city_name} – Last Forecasted PM2.5 (Top {n_geohashes} Geohashes)")
        plt.ylabel("Geohash")
        plt.xlabel("")
        plt.tight_layout()
        plt.show()


    def plot_forecast_time_series(self, num_geohashes: int = 5) -> None:
        subset = list(self.forecasts.items())[:num_geohashes]

        plt.figure(figsize=(14, 6))
        for geohash, df in subset:
            if 'forecast' in df and not df['forecast'].dropna().empty:
                plt.plot(df.index, df['forecast'], label=geohash)

        plt.title(f"{self.city_name} – Forecasted PM2.5 Trends for {num_geohashes} Geohashes")
        plt.xlabel("Time")
        plt.ylabel("PM2.5")
        plt.legend()
        plt.tight_layout()
        plt.show()
