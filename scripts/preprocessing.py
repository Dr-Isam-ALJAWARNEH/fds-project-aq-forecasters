
import pandas as pd
import os
import geohash

class AirQualityPreprocessor:
    def __init__(self, filepath):
        self.filepath = filepath
        self.data = None

    def load_data(self):
        """Load CSV file into a pandas DataFrame."""
        try:
            self.data = pd.read_csv(self.filepath)
            print(f"Data loaded successfully with {self.data.shape[0]} rows and {self.data.shape[1]} columns.")
        except Exception as e:
            print(f"Error loading data: {e}")
            raise

    def basic_cleaning(self):
        """Perform basic cleaning such as datetime conversion and sorting."""
        if self.data is None:
            raise ValueError("Data not loaded. Run load_data() first.")

        # Convert time column from Unix timestamp to datetime
        if 'time' in self.data.columns:
            self.data['datetime'] = pd.to_datetime(self.data['time'], unit='s')
            self.data = self.data.sort_values('datetime').reset_index(drop=True)
            print("Datetime conversion and sorting completed.")
        else:
            raise KeyError("No 'time' column found for datetime conversion.")

    def show_basic_info(self):
        """Display basic information about the dataset."""
        if self.data is None:
            raise ValueError("Data not loaded. Run load_data() first.")

        print(self.data.info())
        print("\nSample Data:\n", self.data.head())

    def handle_optional_columns(self):
        """Handle columns like 'no2' that may or may not exist."""
        if self.data is None:
            raise ValueError("Data not loaded. Run load_data() first.")

        # Example: check if 'no2' exists, if not, skip
        if 'no2' in self.data.columns:
            print("'no2' column exists. Proceeding with 'no2'.")
            # You can add operations related to no2 if needed
        else:
            print("'no2' column missing. Skipping 'no2'-related operations.")

    def add_geohash(self, precision=6):
        """Add geohash encoding from latitude and longitude."""
        if 'latitude' in self.data.columns and 'longitude' in self.data.columns:
            self.data['geohash'] = self.data.apply(
                lambda row: geohash.encode(row['latitude'], row['longitude'], precision=precision), axis=1
            )
            print(f"Geohash added with precision={precision}.")
        else:
            print("Latitude and/or longitude not found. Geohash not added.")

    def save_clean_data(self, output_path):
        """Save the cleaned data to a new CSV file."""
        if self.data is None:
            raise ValueError("No data to save.")
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        self.data.to_csv(output_path, index=False)
        print(f"Cleaned data saved to: {output_path}")
