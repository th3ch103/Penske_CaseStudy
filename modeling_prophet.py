import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf
from prophet import Prophet
from prophet.diagnostics import cross_validation
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.fft import fft, fftfreq
from scipy.signal import find_peaks

def detect_dominant_periods(series, sampling_rate=1, top_n=3, threshold_quantile=0.95):
    """
    Detect dominant periods in a time series using FFT.
    """
    series = series.dropna()
    N = len(series)
    yf = np.abs(fft(series - np.mean(series)))[:N // 2]
    xf = fftfreq(N, d=sampling_rate)[:N // 2]

    threshold = np.quantile(yf, threshold_quantile)
    peaks, _ = find_peaks(yf, height=threshold)

    dominant_freqs = xf[peaks]
    dominant_amps = yf[peaks]
    dominant_periods = 1 / dominant_freqs

    top_indices = np.argsort(dominant_amps)[-top_n:][::-1]

    result = pd.DataFrame({
        'frequency': dominant_freqs[top_indices],
        'period': dominant_periods[top_indices],
        'amplitude': dominant_amps[top_indices]
    })

    return result

#TODO: more flexibility
def prophet_hyperparam_grid_search(
    train_df,
    regressors=None,
    changepoint_scale_list=[0.05],
    prior_scale_list=[0.1],
    fourier_order_grid=None,
    dominant_periods=None,
    seasonality_mode='additive',
    initial_days=None,
    period_days=None,
    horizon_days=None
):
    """
    Perform a grid search over Prophet hyperparameters using cross-validation.
    """
    all_results = []

    # Auto calculate periods
    total_days = (train_df['ds'].max() - train_df['ds'].min()).days
    if horizon_days is None:
        horizon_days = int(total_days * 0.2)
    if initial_days is None:
        initial_days = int(total_days * 0.6)
    if period_days is None:
        period_days = int(total_days * 0.2)

    for cps in changepoint_scale_list:
        for ps in prior_scale_list:
            # If no Fourier specified, only run once with no custom seasonality
            fo_sets = fourier_order_grid if (fourier_order_grid and dominant_periods) else [[None]]

            for fo_list in fo_sets:
                model = Prophet(
                    changepoint_prior_scale=cps,
                    seasonality_mode=seasonality_mode
                )

                # Add regressors if specified
                if regressors:
                    for r in regressors:
                        model.add_regressor(r, prior_scale=ps)

                # Add custom Fourier seasonalities
                if dominant_periods and fo_list[0] is not None:
                    for i, period in enumerate(dominant_periods):
                        model.add_seasonality(name=f'season_{i}', period=period, fourier_order=fo_list[i])

                # Fit
                model.fit(train_df)

                # Cross Validation
                df_cv = cross_validation(
                    model,
                    initial=f'{initial_days} days',
                    period=f'{period_days} days',
                    horizon=f'{horizon_days} days'
                )

                # Clean evaluation
                df_cv = df_cv[df_cv['y'] != 0]
                df_cv['abs_error'] = np.abs(df_cv['yhat'] - df_cv['y'])
                df_cv['squared_error'] = (df_cv['yhat'] - df_cv['y']) ** 2
                df_cv['pct_error'] = np.abs((df_cv['yhat'] - df_cv['y']) / df_cv['y'])

                # Metrics
                mae = df_cv['abs_error'].mean()
                rmse = np.sqrt(df_cv['squared_error'].mean())
                mape = df_cv['pct_error'].mean() * 100

                all_results.append({
                    'changepoint_prior_scale': cps,
                    'prior_scale': ps,
                    'fourier_order_list': fo_list if fo_list[0] is not None else None,
                    'mape': mape,
                    'mae': mae,
                    'rmse': rmse
                })

    results_df = pd.DataFrame(all_results).sort_values(by='mape').reset_index(drop=True)

    best_row = results_df.iloc[0]
    best_config = (
        best_row['changepoint_prior_scale'],
        best_row['prior_scale'],
        best_row['fourier_order_list']
    )

    print("Best Config (cps, prior_scale, [fourier_orders]):", best_config)
    print("Best MAPE: {:.2f}%".format(best_row['mape']))
    print("MAE: {:.2f}, RMSE: {:.2f}".format(best_row['mae'], best_row['rmse']))

    return best_config, results_df
    
def plot_forecast_with_fit(model, train_df, test_df, forecast_df, show_changepoints=True):
    """
    Plot the fitted values and forecasted values from the Prophet model.
    """
    plt.figure(figsize=(14, 6))

    # Plot actual data
    plt.plot(train_df['ds'], train_df['y'], 'k.', label='Train Actual')
    plt.plot(test_df['ds'], test_df['y'], 'ro', label='Test Actual')

    # Fitted values on training data
    forecast_train = model.predict(train_df)
    plt.plot(train_df['ds'], forecast_train['yhat'], label='Fitted (Train)')
    plt.fill_between(train_df['ds'], forecast_train['yhat_lower'], forecast_train['yhat_upper'], alpha=0.2, label='Train CI')

    # Forecast values on test data
    future_test = test_df.copy()
    forecast_test = model.predict(future_test)
    plt.plot(test_df['ds'], forecast_test['yhat'], label='Forecast (Test)')
    plt.fill_between(test_df['ds'], forecast_test['yhat_lower'], forecast_test['yhat_upper'], alpha=0.2, label='Test CI')

    # Vertical line between train/test
    plt.axvline(x=test_df['ds'].min(), color='black', linestyle='--', label='Train/Test Split')

    # Plot changepoints
    if show_changepoints:
        for cp in model.changepoints:
            if cp >= train_df['ds'].min() and cp <= test_df['ds'].max():
                plt.axvline(x=cp, color='grey', linestyle=':', alpha=0.6, label='_nolegend_')

    plt.title("Prophet with Regressors: Fitted vs Forecast vs Actual (with Changepoints)")
    plt.xlabel("Date")
    plt.ylabel("y")
    plt.legend()
    plt.tight_layout()
    plt.show()

def prophet_residual_diagnostics(prophet_model, train_df):
    """
    Perform residual analysis for a fitted Prophet model using a 2x2 subplot:

    """
    # Generate forecasts on the training data
    forecast = prophet_model.predict(train_df)
    
    # Calculate residuals: actual 'y' minus predicted 'yhat'
    residuals = train_df['y'] - forecast['yhat']
    
    # Create a 2x2 subplots figure
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    
    # Top-left: Residuals Over Time
    axs[0, 0].scatter(train_df['ds'], residuals, marker='o', linestyle='-', alpha=0.7)
    axs[0, 0].axhline(0, color='red', linestyle='--')
    axs[0, 0].set_title('Residuals Over Time')
    axs[0, 0].set_xlabel('Date')
    axs[0, 0].set_ylabel('Residuals')
    
    # Top right: Q-Q Plot (using statsmodels.qqplot)
    sm.qqplot(residuals, line='45', fit=True, ax=axs[1, 0])
    axs[0, 1].set_title('Q-Q Plot of Residuals')

    # Bottom-left: Histogram with KDE
    sns.histplot(residuals, kde=True, bins=30, ax=axs[0, 1])
    axs[1, 0].set_title('Histogram & KDE of Residuals')
    axs[1, 0].set_xlabel('Residuals')
    
    # Bottom-right: ACF of Residuals
    plot_acf(residuals.dropna(), lags=30, ax=axs[1, 1], zero=False)
    axs[1, 1].set_title('ACF of Residuals')
    
    fig.tight_layout()
    plt.show()

def evaluate_forecast_performance(y_true, y_pred, verbose=True):
    """
    Evaluate forecast performance using MAE, RMSE, MAPE, and SMAPE.
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    y_true, y_pred = np.array(y_true), np.array(y_pred)
    nonzero_mask = y_true != 0
    mape = np.mean(np.abs((y_true[nonzero_mask] - y_pred[nonzero_mask]) / y_true[nonzero_mask])) * 100

    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    valid = denominator != 0
    smape = (np.mean(np.abs(y_true[valid] - y_pred[valid]) / denominator[valid])) * 100

    if verbose:
        print("Forecast Performance:")
        print(f"MAE   : {mae:.2f}")
        print(f"RMSE  : {rmse:.2f}")
        print(f"MAPE  : {mape:.2f}%")
        print(f"SMAPE : {smape:.2f}%")

    return {'MAE': mae, 'RMSE': rmse, 'MAPE': mape, 'SMAPE': smape}