# Penske_CaseStudy

This case study project focuses on modeling and forecasting time series weekly sales data using ARMA-Based model and Prophet. 
It includes exploratory data analysis (EDA), stationarity checks, univariate and multivariate modeling, parameter tuning, residual diagnostics, and model evaluation using metrics such as MAPE, MAE, and RMSE.

## Project Structure

**Jupyter Notebooks:**
- `1_eda.ipynb`: Exploratory data analysis with stationarity check
- `2_arma_based.ipynb`: ARIMA/SARIMA and ARIMAX/SARIMAX modeling
- `3_prophet.ipynb`: Prophet modeling
  
**Helper Functions:**
- `stationarity.py`: Stationarity test functions
- `modeling_arima.py`: ARIMA modeling functions
- `modeling_prophet.py`: Prophet modeling functions

**Final Model**
- `univariate_prophet.pkl`: Univariate Prophet Model
- `multivariate_prophet.pkl`: Multivariate Propeht Model

## Methodology

### Exploratory Data Analysis (`1_eda.ipynb`):
**1. Data Overview**
- The dataset contains 160 weekly records from Jan 2020 to Jan 2023, including:
  - `weekly_sales`: target variable.
  - `x1_spend`, `x2_spend`: external regressors.
- Initial EDA revealed moderate variance in sales with **no missing values** or **extreme outliers**.
  
**2. Seasonality Analysis**
- **STL decomposition** (with seasonal periods of 7, 13, 26, and 52 weeks):
  - Short-term fluctuations visible with `period=7`, but residuals remain noisy.
  - Longer windows overly smoothed the data, potentially masking signals.
- **Date-based visual inspection**: mild monthly/weekly patterns, but no strong annual/quarterly seasonality.
- **Fourier transform**: no distinct dominant seasonal frequencies observed.
- **ACF/PACF ploat**: no distinct seasonal period observed by `y.diff(s)`.

**3. Stationarity Testing**
- **ADF** and **KPSS** tests confirm that `weekly_sales`, `x1_spend`, and `x2_spend` are all stationary.
- **Rolling statistics** and **ACF/PACF plots** support stationarity visually.
- No significant autocorrelations were observed, suggesting a weak AR/MA structure.
  
**4. Feature & Lag Analysis**
- **Lagged correlation**:
  - `x1_spend`: weak-to-moderate correlation with `weekly_sales`, peaking at lag 0 and lag 25.
  - `x2_spend`: strong immediate effect (lag 0 ~0.80) that quickly diminishes.
- **Multicollinearity**:
  - Low correlation between `x1_spend` and `x2_spend` (~0.02).
  - VIF for both variables = 4.92, below threshold of concern.
  
### ARMA-Based Modeling (`2_arma_based.ipynb`):
This section describes the full modeling process using ARIMA-family models (ARIMA, SARIMA, ARIMAX, SARIMAX), including training, hyperparameter selection, diagnostics, and evaluation.

**1. Preprocessing**  
   - Time-aware 80/20 split using `df.iloc` to preserve sequence.
   - `train_series`: Target variable for model fitting.  
   - `train_exog`: Optional exogenous regressors.

**2. Model Selection via Grid Search**  
   - Hyperparameter tuning via `TimeSeriesSplit` cross-validation.
   - Evaluated using **AIC**, **BIC**, and **MAPE** to identify best model.

**3. Model Training & Forecasting**  
   - Final model fitted on full training data.
   - Forecast made on test set using both target and (if needed) exogenous regressors.

**4. Residual Diagnostics and Performance Evaluation**  
   - Ljung-Box test, ACF/PACF plots, and Q–Q plots ensure white noise residuals and valid assumptions.
   - Forecast accuracy evaluated with MAE, RMSE, MAPE, and SMAPE.

### Prophet Modeling (`3_prophet.ipynb`):
This section outlines the modeling approach using Facebook Prophet, a decomposable time series model that captures trend, seasonality, holidays, and external regressors.

**1. Preprocessing**:
   - Data was reformatted to Prophet’s expected structure (`ds`, `y`).
   - Seasonalities were identified using **Fourier Transform (FFT)** to extract dominant periods (e.g., 2.33, 5.33, 7.11, 8, and 42.67 weeks).
   - Additional exogenous regressors (`x1_spend`, `x2_spend`) were added for multivariate models.

**2. Custom Seasonality**:
   - Prophet’s default weekly/yearly seasonalities were disabled.
   - Custom seasonalities were added based on FFT output using `add_seasonality()`.

**3. Model Selection via Grid Search**:
   - Parameters tuned: `changepoint_prior_scale`, `prior_scale`, and `fourier_orders`.
   - Evaluated using MAPE, MAE, and RMSE on the test set.

**4. Model Training & Forecasting**:
   - Trained on 80% of the data, tested on the final 20%.
     
**4. Residual Diagnostics and Performance Evaluation**  
   - Performance assessed using residual diagnostics.
   - Forecast accuracy evaluated with MAE, RMSE, MAPE, and SMAPE.

## Interpretation

The exploratory data analysis revealed that the weekly sales data is stationary, as confirmed by both the ADF and KPSS tests, and that there is no pronounced seasonality in the STL/FFT decomposition. I start with simple ARIMA models aimed at capturing any short-term autocorrelation inherent in the data without needing to account for strong seasonal components.
| Model     | Order                        | MAPE   | SMAPE  | MAE    | RMSE   |
|-----------|-------------------------------|--------|--------|--------|--------|
| ARIMA     | (0, 0, 1)                     | 36.14% | 25.10% | 1317.3 | 1741.6 | 
| SARIMA    | (0, 0, 1)(0, 0, 1, 7)         | 36.14% | 25.10% | 1317.3 | 1741.6 |
| ARIMAX    | (2, 0, 0) + `x1_spend`, `x2_spend` | 20.34% | 18.23% | 877.4  | 1059.3 |
| SARIMAX   | (2, 0, 0)(0, 0, 0, 7) + `x1_spend`, `x2_spend`  | 20.34% | 18.23% | 877.4  | 1059.3 |

The basic ARIMA and its seasonal variant, SARIMA, yielded similar results with a MAPE of 36.14%, indicating that adding the seasonal component in this case did not enhance model performance given the **absence of dominant seasonal patterns**. To better capture the effects of external factors, the models were extended to ARIMAX and SARIMAX by incorporating the regressors `x1_spend` and `x2_spend`. This integration led to an  improvement in forecasting accuracy, reducing the MAPE to around 20.34%. 

For all models above, the ACF and PACF plots of the residuals showed **no significant lags**, and the Ljung–Box test returned a reasonable p-value, indicating that the **residuals behave like white noise**. Furthermore, the Q–Q plot confirmed that the residuals were approximately normally distributed with **no substantial skewness or kurtosis**, supporting the assumption of well-behaved residuals and the statistical reliability of the forecasts.

Despite the reasonable performance of ARMA-Based model, they rely on **fixed lag structures** and cannot capture **nonlinear effects**, **multiple overlapping seasonalities**, or **sudden changepoints in trend**, thus I moved on with Prophet model.
| Model             | Type         | MAPE   | SMAPE  | MAE    | RMSE   |
|------------------|--------------|--------|--------|--------|--------|
| Prophet           | Univariate   | 33.87% | 23.74% | 1219.4 | 1697.1 |
| Prophet           | Multivariate | 19.55% | 17.55% | 828.4 | 1089.0 |

When deployed in its univariate form, Prophet yielded a MAPE of 33.87%. By integrating the same exogenous variables, the multivariate Prophet model improved the forecast performance marginally further, achieving a MAPE of 19.55%. This **slight performance advantage** compared to ARMA-Based model indicates that Prophet's flexibility in handling nonlinear effects, abrupt changepoints, and non-standard seasonality—even when not strongly pronounced—provides a robust tool for forecasting in such scenarios.

Although Prophet does not formally require residual diagnostics in the same way traditional statistical models do, such checks remain essential in practice. They provide assurance that the fitted model has effectively captured the signal. In the univariate Prophet model, the residual analysis supports the reliability of the results. In the multivariate version, however, while residuals **generally follow a normal distribution**, slight tail deviations in the Q–Q plot suggest mild non-normality. This may indicate that the model underrepresents rare but impactful fluctuations—such as promotional events or supply shocks—that lie outside the range of typical weekly variation.

## Future Work
Building on this, a natural direction for future work is to integrate **deeper domain knowledge**—for instance, through custom holiday effects, event regressors, or categorical segmentation—could improve model expressiveness. More **sophisticated hyperparameter tuning** and more **sophisticated cross-validation** methods would likely enhance generalization performance and reduce variance in real-time deployments.

In addition, exploring **hybrid modeling** approaches that combine the respective strengths of Prophet and traditional ARIMA-based frameworks. Prophet’s adaptability to nonlinearity, irregular seasonality, and structural breaks could be complemented by the statistical rigor and interpretability of ARIMAX or SARIMAX, particularly in capturing autoregressive structure and evaluating model assumptions. Such an ensemble or regime-switching framework may offer a more robust and interpretable solution, especially in scenarios where model transparency is essential for stakeholders.

Another promising line of development lies in the adoption of transformer-based architectures such as **Chronos**, which was released last year and designed specifically to handle temporal patterns with high flexibility.

## Side Note
### How do you sample data
To evaluate forecasting performance while preserving temporal integrity, I applied time series–aware cross-validation methods. For ARMA-Based models, I used `TimeSeriesSplit` from sklearn, which incrementally expands the training window and slides the test window forward. This approach ensures no future data leaks into training and simulates sequential prediction. For Prophet, I used its built-in `cross_validation` function with a rolling-origin strategy, specifying the initial training period, forecast horizon, and step size. This mimics a real-world deployment where the model is retrained regularly and evaluated on unseen future data. These strategies provided consistent and realistic performance assessments across models.

### Why stationary and how to check
Most classical time series models, such as ARIMA, assume that the underlying data is stationary—that is, its statistical properties (mean, variance, autocorrelation) do not change over time. Stationarity is crucial because it ensures that relationships learned from historical data are stable and remain valid in the future. To check for stationarity, I applied two complementary statistical tests using statsmodels in Python:
- **Augmented Dickey–Fuller (ADF) Test**: A unit root test where a significant p-value (typically < 0.05) suggests the series is stationary.
- **Kwiatkowski–Phillips–Schmidt–Shin (KPSS) Test**: Here, a non-significant p-value (typically > 0.05) indicates stationarity.
- In addition to statistical tests, I examined **rolling statistics** (mean and standard deviation) and **time plots** to visually assess whether the series exhibits constant behavior over time.

### What is auto-regressive and Moving Average and how to determine p&q
- An **Auto-Regressive (AR)** model assumes the current value depends on its own previous values. AR(p) model: $y_t = \phi_1 y_{t-1} + \phi_2 y_{t-2} + \dots + \phi_p y_{t-p} + \epsilon_t$
  - \( $p$ \): Number of lag terms  
  - \( $\phi$ \): AR coefficients  
  - \( $\epsilon_t$ \): White noise (random error)

- A **Moving Average (MA)** model uses past forecast errors. MA(q) model: $y_t = \mu + \theta_1 \epsilon_{t-1} + \theta_2 \epsilon_{t-2} + \dots + \theta_q \epsilon_{t-q} + \epsilon_t$
  - \( $q$ \): Number of lagged forecast errors  
  - \( $\theta$ \): MA coefficients  
  - \( $\mu$ \): Mean of the series
  
| Plot  | Purpose | How to Identify |
|-------|---------|-----------------|
| **PACF** | Determines AR (p) | Sharp drop after lag *k* → use p = *k* |
| **ACF**  | Determines MA (q) | Sharp drop after lag *k* → use q = *k* |

### What statistics need to check before presenting model
Before presenting and validating any time series model, several key diagnostics and performance metrics must be evaluated:

**1. Residual Diagnostics**
- **ACF** and **PACF** of residuals: Ensure residuals are uncorrelated and resemble white noise.
- **Ljung–Box test**: Confirms absence of autocorrelation in residuals.
- **Q–Q plot and histogram**: Validate that residuals are approximately normally distributed.
- **Residuals over time**: Check for homoscedasticity and lack of trend/clustering.
  
**2. Model Selection and Fit Metrics**
- **AIC** (Akaike Information Criterion) and **BIC** (Bayesian Information Criterion): Taken model complexity and goodness-of-fit into account.
  
**3. Forecast Evaluation Metrics (on test set)**
- **MAPE** (Mean Absolute Percentage Error): Reflects average forecast error in percentage terms.
- **SMAPE** (Symmetric MAPE): Addresses MAPE’s sensitivity to low values.
- **MAE** (Mean Absolute Error) and **RMSE** (Root Mean Squared Error): Quantify absolute and squared prediction error magnitude.

A reliable model should demonstrate white-noise residuals, good in-sample fit, and low forecast errors on unseen data—validated through both statistical and visual diagnostics.

### Defend final models
Although the time series initially appeared to resemble white noise—exhibiting minimal autocorrelation and no obvious trend or seasonality—Prophet was still a valuable modeling choice. Unlike ARIMA-based models, which rely heavily on clear autocorrelation structures and stationarity, Prophet does not assume a specific underlying time series form. Instead, it offers a flexible additive framework that can still detect weak trends, subtle periodicities, or exogenous influences that may not be visible through traditional statistical diagnostics. Its built-in changepoint detection and intuitive component plots provided greater interpretability and insight—even when working with a seemingly random series.

**Univariate: Prophet**

| Parameter                     | Value         | Rationale                                                                                   |
|------------------------------|---------------|---------------------------------------------------------------------------------------------|
| `seasonality_mode`           | `'multiplicative'` | Additive seasonality assumes a fixed seasonal impact regardless of trend level, which would have underfit the seasonality during high-growth periods. Multiplicative mode more accurately captures this proportional seasonal variation, resulting in a better model fit and more realistic forecasts.|
| `yearly_seasonality`         | `False`       | Yearly component would be noisy/unreliable.          |
| `changepoint_prior_scale`    | `0.1`         | Balances flexibility and overfitting; allows moderate trend changes.                       |
| `n_changepoints`             | `30`          | Capture multiple trend shifts in training data.                    |

| Seasonality Name | Period (Chosen based on FFT)  | Fourier Order | 
|------------------|----------|----------------|
| `7w_season`      | `7.111`  | `6`            |
| `2w_season`      | `2.33`   | `7`            |
| `5w_season`      | `5.333`  | `3`            |
| `42w_season`     | `42.667` | `1`            |
| `8w_season`      | `8`      | `2`            |

**Multivariate: Prophet**

| Parameter                  | Value           | Rationale                                                                 |
|---------------------------|-----------------|--------------------------------------------------------------------------|
| `seasonality_mode`        | `'multiplicative'` | As above       |
| `yearly_seasonality`      | `False`         | As above       |
| `changepoint_prior_scale` | `0.05`          | More conservative trend flexibility to reduce overfitting.               |

| Seasonality Name | Period (Chosen based on FFT)  | Fourier Order |
|------------------|----------|----------------|
| `7w_season`      | `7.11`   | `7`            |
| `2w_season`      | `2.33`   | `6`            |
| `42w_season`     | `42.67`  | `2`            | 
| `8w_season`      | `8.0`    | `1`            | 


## Reference
- [Oracle – Time Series Forecasting (OML4SQL)](https://docs.oracle.com/en/database/oracle/machine-learning/oml4sql/21/dmcon/time-series.html#GUID-0D6954B9-9D66-42E2-A62F-F3FFE84B827E)
- [Statsmodels ARIMA Documentation](https://www.statsmodels.org/stable/generated/statsmodels.tsa.arima.model.ARIMA.html)
- [Interpreting ACF and PACF – Kaggle Notebook by @iamleonie](https://www.kaggle.com/code/iamleonie/time-series-interpreting-acf-and-pacf)
- [Prophet Official Documentation – Python API](https://facebook.github.io/prophet/docs/quick_start.html#python-api)
- [Chronos: Learning the Language of Time Series (arXiv:2403.07815)](https://arxiv.org/html/2403.07815v1)


