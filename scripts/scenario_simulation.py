# scenario_simulation.py

import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict
from forecasting import ARIMAForecastingPipeline
import seaborn as sns

class ScenarioSimulator:
    def __init__(
        self,
        base_df: pd.DataFrame,
        target_col: str,
        city_name: str,
        baseline_forecasts: Dict[str, pd.DataFrame] = None
    ):
        self.base_df = base_df.copy()
        self.target_col = target_col
        self.city_name = city_name
        self.forecasts = baseline_forecasts or {}

    def apply_scenario(self, scenario: str, magnitude: float = 0.2) -> pd.DataFrame:
        df = self.base_df.copy()

        if scenario == "traffic_increase":
            df[self.target_col] *= (1 + magnitude)

        elif scenario == "emission_reduction":
            df[self.target_col] *= (1 - magnitude)

        elif scenario == "wildfire_event":
            if 'time' not in df.columns:
                raise ValueError("DataFrame must contain a 'time' column for wildfire scenario.")
            df['time'] = pd.to_datetime(df['time'])
            mask = df['time'].between('2023-07-01', '2023-07-10')
            df.loc[mask, self.target_col] += magnitude * df[self.target_col].mean()

        else:
            raise ValueError(f"Unknown scenario: {scenario}")

        return df

    def run_simulation(self, scenario: str, magnitude: float = 0.2) -> Dict[str, pd.DataFrame]:
        print(f"\n🚀 Running scenario: {scenario} (magnitude: {magnitude})")
        modified_df = self.apply_scenario(scenario, magnitude)
        scenario_city_name = f"{self.city_name}_{scenario}"

        pipeline = ARIMAForecastingPipeline(modified_df, self.target_col, scenario_city_name)
        pipeline.run()
        return pipeline.forecasts

    def plot_baseline_vs_scenario_comparison(self, scenario_forecasts: Dict[str, pd.DataFrame], ncols: int = 3) -> None:
        common_geohashes = set(self.forecasts.keys()).intersection(scenario_forecasts.keys())
        n = len(common_geohashes)
        if n == 0:
            print("No common geohashes found between baseline and scenario.")
            return

        nrows = (n + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
        axes = axes.flatten()

        for idx, geohash in enumerate(sorted(common_geohashes)):
            baseline_df = self.forecasts[geohash]
            scenario_df = scenario_forecasts[geohash]

            ax = axes[idx]
            ax.plot(baseline_df.index, baseline_df['forecast'], linestyle='--')
            ax.plot(scenario_df.index, scenario_df['forecast'], linestyle='-')
            ax.set_title(f"{geohash}")
            ax.tick_params(axis='x', rotation=45)

        for i in range(idx + 1, len(axes)):
            axes[i].axis('off')

        fig.suptitle(f"{self.city_name} – Baseline vs Scenario Forecasts", fontsize=16)

        # Add single global legend
        fig.legend(["Baseline Forecast", "Scenario Forecast"], loc='upper center', ncol=2)
        plt.tight_layout(rect=[0, 0.05, 1, 0.95])
        plt.show()

    