# EDA and Time-Series Split

from typing import Tuple

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from folium.plugins import HeatMap
import pygeohash as pgh
from IPython.display import display

class AirQualityEDA:
    def __init__(self, df: pd.DataFrame, city_name: str = "City"):
        self.df = df.copy()
        self.city_name = city_name
        if "datetime" in self.df.columns:
            self.df["datetime"] = pd.to_datetime(self.df["datetime"])

    def basic_info(self) -> None:
        print(f"\n--- Basic Information: {self.city_name} ---")
        print(self.df.info())
        print("\n--- Descriptive Statistics ---")
        print(self.df.describe())

    def plot_missing_values(self) -> None:
        plt.figure(figsize=(12, 6))
        missing = self.df.isnull().mean() * 100
        missing = missing[missing > 0]
        missing.sort_values(inplace=True)
        sns.barplot(x=missing.index, y=missing.values)
        plt.title(f"{self.city_name} - % of Missing Values per Column")
        plt.ylabel("% Missing")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    def plot_time_series(self, column: str) -> None:
        if column in self.df.columns:
            plt.figure(figsize=(15, 6))
            plt.plot(self.df['datetime'], self.df[column])
            plt.title(f"{self.city_name} - Time Series of {column}")
            plt.xlabel("Datetime")
            plt.ylabel(column)
            plt.xticks(rotation=45)
            plt.grid()
            plt.tight_layout()
            plt.show()
        else:
            print(f"Column {column} not found in DataFrame.")

    def plot_pm25_by_geohash(self, top_n: int = 5) -> None:
        if 'geohash' not in self.df.columns:
            print("No 'geohash' column found in the data.")
            return

        top_geohashes = self.df['geohash'].value_counts().nlargest(top_n).index
        plt.figure(figsize=(15, 6))

        for geo in top_geohashes:
            subset = self.df[self.df['geohash'] == geo]
            plt.plot(subset['datetime'], subset['pm25'], label=f'Geohash: {geo}')

        plt.title(f"PM2.5 Time Series for Top {top_n} Geohashes")
        plt.xlabel("Datetime")
        plt.ylabel("PM2.5")
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.show()


    def plot_pm25_histogram(self) -> None:
        plt.figure(figsize=(10, 6))
        sns.histplot(self.df['pm25'].dropna(), bins=50, kde=True)
        plt.title(f"{self.city_name} - PM2.5 Distribution")
        plt.xlabel("PM2.5 (µg/m³)")
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_city_comparison(city_data: dict) -> None:
        """
        Plot rolling mean PM2.5 for multiple cities.
        Args:
            city_data: dict of {city_name: DataFrame with 'time' and 'pm25'}
        """
        plt.figure(figsize=(14, 6))

        for city, df in city_data.items():
            df = df.copy()
            df = df.set_index("time").sort_index()
            pm25_ts = df.groupby(df.index)["pm25"].mean()
            pm25_ts.rolling(24).mean().plot(label=city)

        plt.title("Average PM2.5 Over Time by City")
        plt.xlabel("Date")
        plt.ylabel("PM2.5 (µg/m³)")
        plt.legend()
        plt.tight_layout()
        plt.show()

    def display_geohash_heatmap(self, map_center: tuple = None) -> None:
        if 'geohash' not in self.df.columns or 'pm25' not in self.df.columns:
            print("Data must contain 'geohash' and 'pm25' columns.")
            return

        df = self.df.dropna(subset=["pm25", "geohash"])
        agg = df.groupby("geohash")["pm25"].mean().reset_index()
        agg["latlon"] = agg["geohash"].apply(lambda g: pgh.decode(g))
        agg[["lat", "lon"]] = pd.DataFrame(agg["latlon"].tolist(), index=agg.index)
        heat_data = agg[["lat", "lon", "pm25"]].values.tolist()

        if map_center is None:
            map_center = [agg["lat"].mean(), agg["lon"].mean()]

        fmap = folium.Map(location=map_center, zoom_start=11)
        HeatMap(heat_data, radius=10, blur=15, max_zoom=13).add_to(fmap)
        display(fmap)


class TimeSeriesSplit:
    def __init__(self, df: pd.DataFrame, test_size: float = 0.2):
        self.df = df.copy()
        if "datetime" in self.df.columns:
            self.df["datetime"] = pd.to_datetime(self.df["datetime"])
        self.test_size = test_size

    def split(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        self.df = self.df.sort_values("datetime")
        split_idx = int(len(self.df) * (1 - self.test_size))
        train = self.df.iloc[:split_idx]
        test = self.df.iloc[split_idx:]
        print(f"Train size: {train.shape[0]}, Test size: {test.shape[0]}")
        return train, test

    def plot_split(self, column: str) -> None:
        train, test = self.split()
        plt.figure(figsize=(15, 6))
        plt.plot(train['datetime'], train[column], label='Train')
        plt.plot(test['datetime'], test[column], label='Test')
        plt.title(f"Train-Test Split Preview for {column}")
        plt.xlabel("Datetime")
        plt.ylabel(column)
        plt.legend()
        plt.xticks(rotation=45)
        plt.grid()
        plt.show()
