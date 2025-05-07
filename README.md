# 🌍 Air Quality Forecasting with Scenario Simulation

This project forecasts air pollution levels (PM2.5) in New York and Boston using time series modeling (ARIMA), spatial encoding (geohashing), and scenario simulation (e.g., emission reduction, traffic increase). It aims to support data-driven environmental policy planning through interpretable visualizations and forecasting models.

## 📌 Objectives

- Forecast PM2.5 air pollution levels across spatial regions
- Incorporate spatial granularity using geohashes
- Visualize the impact of interventions on future air quality
- Support decision-making through interpretable forecasts

## 🏗️ Project Structure

```python
project/  
│  
├── data/ # Raw and processed datasets  
├── notebooks/  
  ├── preprocessing.ipynb # Data cleaning, geohashing  
  ├── eda.ipynb # Exploratory Data Analysis  
  └── modeling_forecast.ipynb # ARIMA modeling and forecasting    
├── scripts/  
  ├── preprocessing.py # Preprocessing functions  
  ├── eda_and_split.py # EDA and train-test split  
  ├── modeling.py # ARIMA model logic 
  ├── forecasting.py # Creating forecasts based on models
  ├── visualization.py # Visualizing class for forecasts
  └── scenario_simulation.py # ScenarioSimulator class  
├── README.md  
└── requirements.txt
```


## 🔍 Key Features

- **Time Series Forecasting**: Uses ARIMA models per geohash to predict PM2.5 levels.
- **Spatial Analysis**: Spatial segmentation with geohashes allows localized forecasts.
- **Scenario Simulation**: Models interventions like emission reduction.
- **Visualizations**:
  - Forecasts over time
  - Baseline vs. Scenario plots
  - Heatmaps showing PM2.5 improvement by location

## 🛠️ Setup Instructions

1. **Clone the Repository**
   ```bash
   git clone https://github.com/Dr-Isam-ALJAWARNEH/fds-project-aq-forecasters.git
   ```
2. **Open directory**
```bash
  cd air-quality-forecasting
```
3. **Create Virtual Environment**
```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
```
4. **Install Dependencies** 
```bash
pip install -r requirements.txt
```

## How to Run

### Option 1: Run Locally

1.  **Preprocess the data**
    
    -   Run `preprocessing.ipynb` or `scripts/preprocessing.py`
        
2.  **Explore the data**
    
    -   Use `eda.ipynb` to visualize trends, distribution, and geohash insights.
        
3.  **Train Forecasting Models and Simulate Scenarios**
    
    -   Use `modeling_forecast.ipynb` or call `ScenarioSimulator` from `scenario_simulation.py`.

### Option 2: Run on Google Colab

Open the notebook by directly clicking on "open in Colab" badge on the top of the notebooks.

## Example Results

-   Forecast accuracy per region
    
-   Scenario comparisons: emissions down → PM2.5 forecast down
    
-   Heatmaps show spatial impact of policy

## Technologies Used

-   pandas, matplotlib, seaborn, statsmodels, geohash, statsmodels, sklearn

## Authors

- [Nour Eddin Al Shammari](https://github.com/Noureddin-SH)
- [Yonatan Moges](https://github.com/YonatanMoges)




