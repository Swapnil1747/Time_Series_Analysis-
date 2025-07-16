# Time Series Analysis GUI

This is a Streamlit-based interactive GUI application for performing time series analysis on two datasets: `AirPassengers.csv` and `dataset.txt`. The app provides various visualization and analysis tools to explore time series data, including decomposition, detrending, autocorrelation, and Granger causality testing.

## Features

- Load and display datasets (`AirPassengers.csv` and `dataset.txt`).
- Line plot and fill-between plot for visualizing time series.
- Seasonal decomposition (additive and multiplicative models).
- Detrending using least squares fit or trend component subtraction.
- Deseasonalizing the time series.
- Autocorrelation plot.
- ACF (Autocorrelation Function) and PACF (Partial Autocorrelation Function) plots.
- Lag plots for visualizing lag relationships.
- Granger causality test on the `dataset.txt` data.

## Installation

1. Clone the repository or download the source code.
2. Create a virtual environment (recommended):

```bash
python -m venv venv
```

3. Activate the virtual environment:

- On Windows:

```bash
venv\Scripts\activate
```

- On macOS/Linux:

```bash
source venv/bin/activate
```

4. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

Run the Streamlit app with the following command:

```bash
streamlit run streamlit_app.py
```

The app will open in your default web browser. Use the sidebar to select datasets and analysis options.

## Files

- `streamlit_app.py`: Main Streamlit application script.
- `AirPassengers.csv`: Dataset containing monthly airline passenger numbers.
- `dataset.txt`: Dataset used for Granger causality testing.

## Dependencies

- streamlit
- pandas
- numpy
- matplotlib
- statsmodels
- scipy

## License

This project is licensed under the MIT License.
