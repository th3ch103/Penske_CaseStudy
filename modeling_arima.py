import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.stats.diagnostic import acorr_ljungbox
from scipy.stats import probplot

def train_and_forecast_arima(train_data, test_data, order, seasonal_order=None, exog_train=None, exog_test=None):
    """
    Train an ARIMA/SARIMA model on train_data and forecast on test_data.
    Returns the fitted model, forecast values, and AIC/BIC.
    """
    model = SARIMAX(
        train_data, 
        order=order, 
        seasonal_order=seasonal_order if seasonal_order else (0, 0, 0, 0),
        exog=exog_train,
        trend='c'
    )
    fitted_model = model.fit(disp=False)
    
    # Forecast the length of the test set
    forecast = fitted_model.forecast(steps=len(test_data), exog=exog_test)
    
    return fitted_model, forecast, fitted_model.aic, fitted_model.bic

# def smape(a, f):
#     """
#     Symmetric Mean Absolute Percentage Error.
#     """
#     a, f = np.array(a), np.array(f)
#     denominator = (np.abs(a) + np.abs(f))
#     # Avoid division by zero
#     mask = denominator != 0
#     return 100 * np.mean(2.0 * np.abs(f[mask] - a[mask]) / denominator[mask])

def mean_absolute_percentage_error(y_true, y_pred):
    """
    Compute MAPE (Mean Absolute Percentage Error) between two arrays.
    Returns a float representing the percentage error.
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    # To avoid division by zero, filter out zeros (if any)
    non_zero_idx = y_true != 0
    return np.mean(np.abs((y_true[non_zero_idx] - y_pred[non_zero_idx]) / y_true[non_zero_idx])) * 100

def grid_search_arima_cv(series, param_grid, seasonal_param_grid=None, n_splits=3, exog=None):
    """
    Perform a grid search over ARIMA orders (and seasonal orders if provided) 
    using Time Series cross-validation. For each fold, we compute average AIC, BIC, and MAPE.
    """
    import itertools
    
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    # Generate all (p,d,q) combinations
    pdq = list(itertools.product(param_grid['p'], param_grid['d'], param_grid['q']))
    
    # If seasonal parameters are provided, generate (P,D,Q,m) combinations; otherwise [None]
    if seasonal_param_grid:
        seasonal_pdq = list(itertools.product(
            seasonal_param_grid['P'],
            seasonal_param_grid['D'],
            seasonal_param_grid['Q'],
            seasonal_param_grid['m']
        ))
    else:
        seasonal_pdq = [None]
    
    results = []
    
    # Iterate over all combinations of pdq and seasonal_pdq
    for order in pdq:
        for s_order in seasonal_pdq:
            
            # Lists to store metrics for each fold
            fold_aics, fold_bics, fold_mapes = [], [], []
            
            # Cross-validation
            for train_index, test_index in tscv.split(series):
                train_data = series.iloc[train_index]
                test_data = series.iloc[test_index]
                
                # If exog is provided, split exog accordingly
                if exog is not None:
                    exog_train = exog.iloc[train_index]
                    exog_test = exog.iloc[test_index]
                else:
                    exog_train, exog_test = None, None

                try:
                    model_fit, forecast, aic, bic = train_and_forecast_arima(
                        train_data, 
                        test_data, 
                        order=order, 
                        seasonal_order=s_order, 
                        exog_train=exog_train, 
                        exog_test=exog_test
                    )
                    fold_aics.append(aic)
                    fold_bics.append(bic)
                    
                    # Compute MAPE
                    mape_val = mean_absolute_percentage_error(test_data, forecast)
                    fold_mapes.append(mape_val)
                except Exception as e:
                    # If model fails to converge, skip
                    # You could also log this error
                    break
            
            # If we successfully got metrics for all folds, compute averages
            if len(fold_aics) == n_splits:
                avg_aic = np.mean(fold_aics)
                avg_bic = np.mean(fold_bics)
                avg_mape = np.mean(fold_mapes)
                
                results.append({
                    'order': order,
                    'seasonal_order': s_order,
                    'avg_aic': avg_aic,
                    'avg_bic': avg_bic,
                    'avg_mape': avg_mape
                })
    
    results_df = pd.DataFrame(results)
    
    # Sort results by avg_mape (ascending) or by avg_aic, etc. 
    results_df.sort_values(by='avg_mape', ascending=True, inplace=True)
    
    return results_df

def plot_forecast_vs_actual(train_series, test_series, forecast_series, model_order):
    """
    Plot the train data, test data, and forecast.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(train_series.index, train_series, label='Train')
    plt.plot(test_series.index, test_series, label='Test', color='orange')
    plt.plot(forecast_series.index, forecast_series, label='Forecast', color='green', linestyle='--')
    
    plt.title(f"ARIMA{model_order} Forecast vs Actual")
    plt.xlabel("Date")
    plt.ylabel("Value")
    plt.legend()
    plt.show()

def residual_diagnostics(model_fit, model_order="ARIMA", lags=30):
    """
    Generate a 2x2 residual diagnostic plot and print Ljung-Box test results.
    """
    # Extract and clean fitted values and residuals
    fittedvalues = model_fit.fittedvalues.dropna()
    residuals = model_fit.resid.dropna()

    # Create a 2x2 figure for diagnostics
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f"Residual Diagnostics for {model_order}", fontsize=14)

    # Top-left: Residuals over time
    axs[0, 0].scatter(fittedvalues, residuals)
    axs[0, 0].axhline(0, color='red', linestyle='--')
    axs[0, 0].set_title("Residuals vs Fitted Values")
    axs[0, 0].set_xlabel("Fitted Values")
    axs[0, 0].set_ylabel("Residuals")

    # Top-right: Q-Q Plot for normality
    qq_data = probplot(residuals, dist="norm")
    axs[0, 1].scatter(qq_data[0][0], qq_data[0][1])
    slope, intercept = qq_data[1][0], qq_data[1][1]
    x_vals = np.linspace(min(qq_data[0][0]), max(qq_data[0][0]), 100)
    axs[0, 1].plot(x_vals, slope * x_vals + intercept, color='red')
    axs[0, 1].set_title("Q–Q Plot")

    # Bottom-left: ACF plot of residuals
    plot_acf(residuals, ax=axs[1, 0], lags=lags, zero=False)
    axs[1, 0].set_title("ACF of Residuals")

    # Bottom-right: PACF plot of residuals using the 'ywm' method
    plot_pacf(residuals, ax=axs[1, 1], lags=lags, zero=False, method='ywm')
    axs[1, 1].set_title("PACF of Residuals")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

    # Ljung-Box Test for autocorrelation in residuals
    lb_test = acorr_ljungbox(residuals, lags=[lags], return_df=True)
    print(f"Ljung-Box test for {model_order} (lag={lags}):")
    print(lb_test)


def forecast_performance(y_true, y_pred, model_order="ARIMA"):
    """
    Compute and print forecast performance metrics: MAE, RMSE, MAPE, and SMAPE.
    """
    # Compute Mean Absolute Error
    mae = mean_absolute_error(y_true, y_pred)
    
    # Compute Root Mean Squared Error
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    # Compute Mean Absolute Percentage Error (MAPE)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    
    # Compute Symmetric MAPE (SMAPE)
    denominator = (np.abs(y_true) + np.abs(y_pred))
    mask = denominator != 0
    smape = 100 * np.mean(2.0 * np.abs(y_pred[mask] - y_true[mask]) / denominator[mask])
    
    # Print the performance metrics
    print(f"Forecast Performance for {model_order}:")
    print(f"MAE   : {mae:.2f}")
    print(f"RMSE  : {rmse:.2f}")
    print(f"MAPE  : {mape:.2f}%")
    print(f"SMAPE : {smape:.2f}%")
    
    # Return a dictionary of metrics
    performance = {
        'MAE': mae,
        'RMSE': rmse,
        'MAPE': mape,
        'SMAPE': smape
    }
    return performance


#TODO: seasonal order doesnt show up on plot