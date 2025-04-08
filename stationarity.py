import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tsa.seasonal import STL
from scipy.fft import fft, fftfreq


def plot_date_based(df):
    """
    Plot boxplots of target data based on month, quarter, week of year, and year.
    """
    df_plot = df.copy()
    df_plot['month'] = df_plot.index.month
    df_plot['quarter'] = df_plot.index.quarter
    df_plot['weekofyear'] = df_plot.index.isocalendar().week
    df_plot['year'] = df_plot.index.year

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    sns.boxplot(x='month', y='weekly_sales', data=df_plot, ax=axes[0, 0])
    axes[0, 0].set_title('Sales by Month')

    sns.boxplot(x='quarter', y='weekly_sales', data=df_plot, ax=axes[0, 1])
    axes[0, 1].set_title('Sales by Quarter')

    sns.boxplot(x='weekofyear', y='weekly_sales', data=df_plot, ax=axes[1, 0])
    axes[1, 0].set_title('Sales by Week of Year')
    axes[1, 0].tick_params(axis='x', rotation=90)

    sns.boxplot(x='year', y='weekly_sales', data=df_plot, ax=axes[1, 1])
    axes[1, 1].set_title('Sales by Year')

    plt.tight_layout()
    plt.show()

def plot_rolling_statistics(series, window=12):
    """
    Plot rolling mean and rolling standard deviation to visually assess stationarity.
    """
    rolmean = series.rolling(window=window).mean()
    rolstd = series.rolling(window=window).std()

    plt.figure(figsize=(12,6))
    plt.plot(series, label='Original')
    plt.plot(rolmean, color='red', label=f'Rolling Mean (window={window})', linestyle='--')
    plt.plot(rolstd, color='green', label=f'Rolling Std (window={window})', linestyle='--')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show()


def check_stationarity(series, alpha=0.05, verbose=True):
    """
    Check if a time series is stationary using the Augmented Dickey-Fuller (ADF) and KPSS tests.
    """

    # Remove missing values from the series
    series_clean = series.dropna()

    # 1. Augmented Dickey-Fuller Test
    adf_result = adfuller(series_clean, autolag='AIC')
    adf_stat = adf_result[0]
    adf_pvalue = adf_result[1]
    adf_crit_vals = adf_result[4]
    # For ADF, the null hypothesis is non-stationarity.
    # If p-value < alpha, we reject the null hypothesis and assume the series is stationary.
    adf_stationary = adf_pvalue < alpha

    # 2. KPSS Test
    # Using regression='c' assumes a constant (i.e., level) stationarity.
    kpss_result = kpss(series_clean, regression='c', nlags='auto')
    kpss_stat = kpss_result[0]
    kpss_pvalue = kpss_result[1]
    kpss_crit_vals = kpss_result[3]
    # For KPSS, the null hypothesis is stationarity.
    # If p-value >= alpha, we fail to reject the null hypothesis and assume the series is stationary.
    kpss_stationary = kpss_pvalue >= alpha

    # Print detailed test results if verbose is True
    if verbose:
        print(f"ADF Statistic: {adf_stat:.4f}")
        print(f"p-value: {adf_pvalue:.4f}")
        print("Critical Values:")
        for key, value in adf_crit_vals.items():
            print(f"   {key}: {value:.4f}")
        print("Conclusion: The series is", "stationary." if adf_stationary else "non-stationary.")
        print(f"KPSS Statistic: {kpss_stat:.4f}")
        print(f"p-value: {kpss_pvalue:.4f}")
        print("Critical Values:")
        for key, value in kpss_crit_vals.items():
            print(f"   {key}: {value:.4f}")
        print("Conclusion: The series is", "stationary." if kpss_stationary else "non-stationary.")


def plot_acf_pacf(series, lags=30, seasonal_lags=None):
    """
    Plot the Autocorrelation (ACF) and Partial Autocorrelation (PACF) of a time series
    """
    import matplotlib.pyplot as plt
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

    # Drop missing values for accurate analysis.
    series_clean = series.dropna()

    # Create a figure with two subplots: one for ACF and one for PACF.
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    # Plot ACF on the left subplot.
    plot_acf(series_clean, lags=lags, ax=axes[0], zero=False)
    axes[0].set_title('Autocorrelation Function (ACF)')

    # Plot PACF on the right subplot using the 'ywm' method.
    plot_pacf(series_clean, lags=lags, ax=axes[1], method='ywm', zero=False)
    axes[1].set_title('Partial Autocorrelation Function (PACF)')

    # If seasonal lags are provided, draw vertical lines on both plots.
    if seasonal_lags is not None:
        for lag in seasonal_lags:
            if lag <= lags:
                axes[0].axvline(x=lag, color='red', linestyle='--', alpha=0.7)
                axes[0].text(lag+0.5, 0.7, f'Lag {lag}', color='red')
                axes[1].axvline(x=lag, color='red', linestyle='--', alpha=0.7)
                axes[1].text(lag+0.5, 0.7, f'Lag {lag}', color='red')

    # Adjust layout and display the plots.
    plt.tight_layout()
    plt.show()

def stl_decomposition(series, period):
    """
    Perform STL decomposition on a time series and plot the trend, seasonal, and residual components.
    """
    # Drop any missing values from the series to ensure proper decomposition
    series_clean = series.dropna()

    # Initialize the STL object with the specified period and set robust=True to mitigate the influence of outliers
    stl = STL(series_clean, period=period, robust=True)
    result = stl.fit()

    # Plot the decomposition results
    fig = result.plot()
    fig.set_size_inches(7, 5)
    plt.show()

    return result

def fourier_decomposition(series, sampling_rate=1):
    """
    Performs FFT spectrum analysis and plots the amplitude vs frequency.
    """
    # Step 1: Prepare data
    y = series.dropna().values
    N = len(y)

    # Step 2: FFT transform
    fft_vals = fft(y)
    fft_magnitude = 2.0 / N * np.abs(fft_vals[:N // 2])
    freq = fftfreq(N, d=1 / sampling_rate)[:N // 2]

    # Step 3: Plot
    plt.figure(figsize=(12, 5))
    plt.stem(freq, fft_magnitude, basefmt=" ")
    plt.title("FFT - Frequency Spectrum")
    plt.xlabel("Frequency")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_lagged_correlation(exog_series, y_series, max_lag=14, exog_name="x1"):
    """
    Plot the lagged correlation between an exogenous variable and the target variable.
    """
    corrs = []
    for lag in range(0, max_lag + 1):
        shifted = exog_series.shift(lag)
        corr = shifted.corr(y_series)
        corrs.append(corr)

    plt.figure(figsize=(8, 4))
    plt.plot(range(0, max_lag + 1), corrs, marker='o')
    plt.title(f"Lagged Correlation of {exog_name} vs Target")
    plt.xlabel("Lag")
    plt.ylabel("Correlation with y")
    plt.grid(True)
    plt.xticks(range(0, max_lag + 1))
    plt.legend()
    plt.show()

    return pd.Series(corrs, index=range(0, max_lag + 1))

def check_multicollinearity(df, exog_cols):
    """
    Check for multicollinearity using Variance Inflation Factor (VIF) and plot correlation heatmap.
    """
    X = df[exog_cols].dropna()
    vif_df = pd.DataFrame()
    vif_df["feature"] = exog_cols
    vif_df["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    vif_df = vif_df.set_index("feature").sort_values("VIF", ascending=False)
    print(vif_df)

    # Correlation heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(df[exog_cols].corr(), annot=True, cmap='coolwarm', center=0)
    plt.title("Correlation Between Variables")
    plt.tight_layout()
    plt.show()

    return vif_df

def inverse_boxcox(y, lambda_):
    if lambda_ == 0:
        return np.expm1(y)
    else:
        return np.expm1(np.log(lambda_ * y + 1) / lambda_)