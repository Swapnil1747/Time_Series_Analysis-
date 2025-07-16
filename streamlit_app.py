import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from pandas.plotting import lag_plot, autocorrelation_plot
from statsmodels.tsa.stattools import grangercausalitytests
from scipy import signal

st.title("Time Series Analysis GUI")

@st.cache_data
def load_airpassengers():
    df = pd.read_csv("AirPassengers.csv")
    df.columns = ["Date", "Number of Passengers"]
    return df

@st.cache_data
def load_dataset_txt():
    data = pd.read_csv("dataset.txt")
    data['date'] = pd.to_datetime(data['date'])
    data['month'] = data.date.dt.month
    return data

def plot_line(df, x, y, title, xlabel, ylabel):
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df[x], df[y], color="tab:red")
    ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
    st.pyplot(fig)

def plot_fill_between(df):
    x = df["Date"]
    y1 = df["Number of Passengers"].values
    fig, ax = plt.subplots(figsize=(16, 5))
    ax.fill_between(x, y1=y1, y2=-y1, alpha=0.5, linewidth=2, color="seagreen")
    ax.set_title('Air Passengers (Two Side View)', fontsize=16)
    ax.hlines(y=0, xmin=np.min(df['Date']), xmax=np.max(df['Date']), linewidth=.5)
    st.pyplot(fig)

def plot_decomposition(df, model):
    result = seasonal_decompose(df['Number of Passengers'], model=model, period=30)
    fig = result.plot()
    fig.set_size_inches(16, 12)
    st.pyplot(fig)

def plot_detrended(df):
    detrended = signal.detrend(df['Number of Passengers'].values)
    fig, ax = plt.subplots()
    ax.plot(detrended)
    ax.set_title('Air Passengers detrended by subtracting the least squares fit')
    st.pyplot(fig)

def plot_detrended_trend_component(df):
    result_mul = seasonal_decompose(df['Number of Passengers'], model='multiplicative', period=30)
    detrended = df['Number of Passengers'].values - result_mul.trend
    fig, ax = plt.subplots()
    ax.plot(detrended)
    ax.set_title('Air Passengers detrended by subtracting the trend component')
    st.pyplot(fig)

def plot_deseasonalized(df):
    result_mul = seasonal_decompose(df['Number of Passengers'], model='multiplicative', period=30)
    deseasonalized = df['Number of Passengers'].values / result_mul.seasonal
    fig, ax = plt.subplots()
    ax.plot(deseasonalized)
    ax.set_title('Air Passengers Deseasonalized')
    st.pyplot(fig)

def plot_autocorrelation(df):
    fig, ax = plt.subplots()
    autocorrelation_plot(df['Number of Passengers'], ax=ax)
    st.pyplot(fig)

def plot_acf_pacf(df):
    fig, axes = plt.subplots(1, 2, figsize=(16, 3))
    plot_acf(df['Number of Passengers'], lags=50, ax=axes[0])
    plot_pacf(df['Number of Passengers'], lags=50, ax=axes[1])
    st.pyplot(fig)

def plot_lag(df):
    fig, axes = plt.subplots(1, 4, figsize=(10, 3), sharex=True, sharey=True)
    for i, ax in enumerate(axes.flatten()[:4]):
        lag_plot(df['Number of Passengers'], lag=i+1, ax=ax, c='firebrick')
        ax.set_title('Lag ' + str(i+1))
    fig.suptitle('Lag Plots of Air Passengers', y=1.05)
    st.pyplot(fig)

def granger_causality_test(data):
    st.write("Granger Causality Test Results:")
    maxlag = st.slider("Select max lag for Granger Causality Test", 1, 5, 2)
    test_result = grangercausalitytests(data[['value', 'month']], maxlag=maxlag, verbose=True)
    st.write("Test completed. See console for detailed output.")

def main():
    st.sidebar.title("Options")
    data_source = st.sidebar.selectbox("Select Dataset", ["AirPassengers.csv", "dataset.txt"])
    
    if data_source == "AirPassengers.csv":
        df = load_airpassengers()
        st.subheader("Air Passengers Data")
        st.dataframe(df)
        
def main():
    st.sidebar.title("Options")
    data_source = st.sidebar.selectbox("Select Dataset", ["AirPassengers.csv", "dataset.txt"])
    
    if data_source == "AirPassengers.csv":
        df = load_airpassengers()
        st.subheader("Air Passengers Data")
        st.dataframe(df)
        
        if st.sidebar.button("Show Line Plot"):
            st.subheader("Line Plot")
            plot_line(df, "Date", "Number of Passengers", "Air Passengers Over Time", "Date", "Number of Passengers")

        if st.sidebar.button("Show Fill Between Plot"):
            st.subheader("Fill Between Plot")
            plot_fill_between(df)

        st.sidebar.subheader("Decomposition")
        model = st.sidebar.selectbox("Select Decomposition Model", ["multiplicative", "additive"])
        if st.sidebar.button("Show Decomposition Plot"):
            plot_decomposition(df, model)

        st.sidebar.subheader("Detrending")
        detrend_option = st.sidebar.selectbox("Detrending Method", ["Least Squares Fit", "Trend Component"])
        if st.sidebar.button("Show Detrending Plot"):
            if detrend_option == "Least Squares Fit":
                plot_detrended(df)
            else:
                plot_detrended_trend_component(df)

        if st.sidebar.button("Show Deseasonalizing Plot"):
            st.subheader("Deseasonalizing")
            plot_deseasonalized(df)

        if st.sidebar.button("Show Autocorrelation Plot"):
            st.subheader("Autocorrelation Plot")
            plot_autocorrelation(df)

        if st.sidebar.button("Show ACF and PACF Plots"):
            st.subheader("ACF and PACF Plots")
            plot_acf_pacf(df)

        if st.sidebar.button("Show Lag Plots"):
            st.subheader("Lag Plots")
            plot_lag(df)
        
    else:
        data = load_dataset_txt()
        st.subheader("Dataset for Granger Causality Test")
        st.dataframe(data)
        granger_causality_test(data)

if __name__ == "__main__":
    main()
