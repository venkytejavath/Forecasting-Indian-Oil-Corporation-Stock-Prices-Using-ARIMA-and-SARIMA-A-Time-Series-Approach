# Forecasting Indian Oil Corporation Stock Prices Using ARIMA and SARIMA

## 📌 Project Overview

This project performs **time-series analysis and stock-price forecasting
of Indian Oil Corporation (IOC.NS)** using statistical forecasting
models, primarily **ARIMA and SARIMA**.

The analysis follows a complete forecasting workflow:

**Data Collection → Data Cleaning → Exploratory Data Analysis →
Stationarity Testing → Differencing → ACF/PACF Analysis → Seasonal
Decomposition → Train/Validation/Test Split → ARIMA Model Selection →
Naïve Benchmark → SARIMA Model Selection → Residual Diagnostics → Model
Selection → Future Forecasting**

The main objective is to investigate how well ARIMA-family models can
capture the temporal structure of IOC's historical closing prices and
whether they provide an improvement over a simple naïve random-walk
forecast.

------------------------------------------------------------------------

## 🎯 Objectives

The project aims to:

-   Collect historical IOC stock-price data.
-   Clean and prepare the time-series data.
-   Explore the historical behavior and variability of IOC closing
    prices.
-   Test the stationarity of the price series.
-   Apply first-order differencing when required.
-   Examine autocorrelation using ACF and PACF.
-   Explore possible weekly seasonality using a seasonal period of 5
    trading observations.
-   Build and compare ARIMA models using walk-forward validation.
-   Establish a naïve random-walk model as a baseline.
-   Build and compare SARIMA models.
-   Evaluate models using MAE and RMSE.
-   Perform residual diagnostics using Ljung--Box and ARCH tests.
-   Select the final model using validation performance while keeping
    the test set separate.
-   Generate a 30-step future forecast with 95% confidence intervals.

------------------------------------------------------------------------

## 📊 Dataset

### Source

Historical market data are downloaded using the `yfinance` Python
library for the Yahoo Finance ticker:

``` text
IOC.NS
```

The notebook requests data from:

``` text
Start: 2010-01-01
End:   2026-01-01
```

with `auto_adjust=True`.

Only the following variables are retained:

-   `Date`
-   `Close`

The cleaned dataset is saved as:

``` text
IOC_NS_stock_data.csv
```

The notebook also checks for:

-   Missing values
-   Duplicate rows
-   Duplicate dates
-   Correct date format
-   Chronological ordering

------------------------------------------------------------------------

## 🔬 Exploratory Data Analysis

The project includes several exploratory analyses of IOC's closing-price
series:

### Historical price trend

The historical closing-price plot is used to examine:

-   Long-term trends
-   Major price movements
-   Periods of sharp increases or decreases

### Distribution analysis

The project uses:

-   Histogram
-   Kernel Density Estimate (KDE)
-   Box plot

to examine the distribution and spread of closing prices.

### Rolling statistics

A **30-observation rolling mean** and **rolling standard deviation** are
calculated.

These help examine:

-   Local price trends
-   Changes in variability
-   Periods of higher and lower volatility

------------------------------------------------------------------------

## 📈 Stationarity Analysis

The **Augmented Dickey-Fuller (ADF) test** is used to assess whether the
original closing-price series is stationary.

The original price series has a high ADF p-value of approximately:

``` text
0.9871
```

Therefore, there is insufficient evidence to reject the null hypothesis
of a unit root, indicating that the original price series is
non-stationary.

First-order differencing is then applied:

``` python
close_diff = IOC["Close"].diff().dropna()
```

The differenced series is found to be stationary according to the ADF
test.

The notebook also calculates log returns:

``` python
IOC["Log_Return"] = np.log(
    IOC["Close"] / IOC["Close"].shift(1)
)
```

The log returns are calculated for examining relative day-to-day price
changes.

------------------------------------------------------------------------

## 🔁 ACF and PACF Analysis

ACF and PACF plots are generated for the first-differenced series.

They are used to investigate the autocorrelation structure and provide
guidance for selecting possible ARIMA parameters:

-   `p` --- autoregressive order
-   `d` --- differencing order
-   `q` --- moving-average order

------------------------------------------------------------------------

## 📅 Seasonal Analysis

A seasonal period of:

``` text
s = 5
```

is considered because there are approximately **five trading
observations in a typical trading week**.

This is treated as an **exploratory seasonal hypothesis**, not as an
assumption that IOC necessarily has deterministic weekly seasonality.

Seasonal decomposition is used to explore whether a repeating pattern
exists at this period.

The same seasonal period is considered during SARIMA modelling.

------------------------------------------------------------------------

## 🧪 Train--Validation--Test Strategy

The dataset is divided chronologically into:

-   **Training set:** used for fitting candidate models
-   **Validation set:** used for model selection
-   **Test set:** used only for final performance evaluation

The notebook uses:

``` text
Validation observations = 60
Test observations       = 60
```

The chronological structure is preserved to prevent future information
from being used during model selection.

This is particularly important for time-series forecasting because
randomly shuffling observations would introduce look-ahead bias.

------------------------------------------------------------------------

## 📏 Evaluation Metrics

Two forecasting metrics are used:

### Mean Absolute Error (MAE)

\[ MAE =
`\frac{1}{n}`{=tex}`\sum`{=tex}\_{i=1}\^{n}\|y_i-`\hat{y}`{=tex}\_i\| \]

MAE measures the average absolute difference between actual and
predicted prices.

### Root Mean Squared Error (RMSE)

\[ RMSE = `\sqrt{
\frac{1}{n}
\sum_{i=1}^{n}
(y_i-\hat{y}_i)^2
}`{=tex} \]

RMSE gives greater weight to larger forecasting errors.

Lower MAE and RMSE indicate better forecasting performance.

------------------------------------------------------------------------

## 🤖 ARIMA Modelling

Candidate ARIMA models are evaluated using walk-forward validation.

The search considers:

``` text
p = range(0,0)
d = [0, 1]
q = (0,3)
```

Each candidate model forecasts the validation observations one step at a
time.

After each prediction, the actual observation is added to the history
before forecasting the next observation.

This produces a realistic one-step-ahead forecasting evaluation.

### Best ARIMA Model

The best model selected using validation RMSE is:

``` text
ARIMA(1,1,1)
```

Validation performance:

``` text
MAE  ≈ 1.1455
RMSE ≈ 1.5089
```

The model-selection result reported by the notebook identifies
ARIMA(1,1,1) as the best ARIMA specification.

------------------------------------------------------------------------

## 🧍 Naïve Random-Walk Benchmark

A naïve forecasting model is included as a baseline.

For each future observation, the naïve model predicts the most recently
observed price.

This provides an important benchmark because stock prices can be
difficult to forecast and a sophisticated model should demonstrate
improvement over a simple random-walk approach.

### Validation

``` text
Naïve RMSE ≈ 1.5166
ARIMA RMSE ≈ 1.5089
```

ARIMA provides a small improvement over the naïve benchmark.

### Test

``` text
Naïve RMSE ≈ 2.0474
ARIMA RMSE ≈ 2.0399
```

Again, ARIMA performs slightly better.

------------------------------------------------------------------------

## 📊 SARIMA Modelling

SARIMA models are evaluated using walk-forward validation with:

``` text
Seasonal period = 5
```

The candidate parameter grid contains:

``` text
p = [0, 1]
d = [1]
q = [0, 1]

P = [0, 1]
D = [0]
Q = [0, 1]

s = 5
```

This produces 16 candidate SARIMA specifications.

Models that fail to converge or do not produce the required number of
forecasts are skipped.

### Best SARIMA Model

The model selected by validation RMSE is:

``` text
SARIMA(1,1,1)(1,0,1,5)
```

Validation performance reported in the notebook:

``` text
MAE  ≈ 1.1405
RMSE ≈ 1.5157
```

The SARIMA model therefore has a slightly lower validation RMSE than
ARIMA.

------------------------------------------------------------------------

## 🏆 Final Model Selection

Importantly, the **test set is not used to select the final model**.

The final decision is based only on validation RMSE:

``` text
ARIMA validation RMSE  ≈ 1.5089
SARIMA validation RMSE ≈ 1.5078
```

Therefore, the notebook selects:

``` text
SARIMA(1,1,1)(1,0,1,5)
```

as the final model for future forecasting.

This preserves the test set as an unseen evaluation dataset.

------------------------------------------------------------------------

## 📋 Final Test Results

The final test-set comparison is:

  Model                         MAE         RMSE
  ------------------------ -------- ------------
  ARIMA(1,1,1)               1.4752   **2.0399**
  SARIMA(1,1,1)(1,0,1,5)     1.4763       2.0439
  Naïve Random Walk          1.4874       2.0474

### Interpretation

The models perform very similarly.

ARIMA achieves the lowest test RMSE by a very small margin, while SARIMA
was selected as the final model because it achieved the lowest
**validation** RMSE.

This distinction is important:

> **The final model is selected using validation performance, not by
> choosing the model with the lowest test error.**

The results suggest that the ARIMA-family models provide only a
**marginal improvement over the naïve random-walk benchmark** for this
forecasting task.

------------------------------------------------------------------------

## 🩺 Residual Diagnostics

Residual diagnostics are performed to determine whether the final model
has adequately captured the time-series structure.

The notebook examines:

-   Residual time series
-   Residual distribution
-   Residual ACF
-   Ljung--Box test
-   ARCH test
-   Q-Q plot

### Ljung--Box Test

The Ljung--Box results show:

-   No significant autocorrelation at lags 5 and 10
-   Significant autocorrelation at lags 20 and 30

This suggests that some temporal dependence remains in the residuals.

### ARCH Test

The ARCH test produces a very small p-value, providing strong evidence
of **heteroskedasticity**.

This means that residual variance changes over time.

The finding suggests that a volatility model such as **GARCH** could be
investigated in future work.

### Q-Q Plot

The residual Q-Q plot shows deviations from the normal reference line,
particularly in the tails.

This indicates that the residuals are not perfectly normally distributed
and contain some extreme observations.

------------------------------------------------------------------------

## 🔮 Future Forecasting

The selected model is refitted using the complete available IOC dataset
and used to generate:

``` text
30 future forecast observations
```

The forecast includes:

-   Point forecast
-   Lower 95% confidence bound
-   Upper 95% confidence bound

The notebook visualizes the forecast together with recent historical
prices.

The confidence interval represents uncertainty around the predicted
values, and the uncertainty generally increases farther into the
forecasting horizon.

> **Note:** The notebook currently generates future labels using
> `pandas.bdate_range()`, which represents Monday--Friday business days.
> For a production implementation, these dates should ideally be mapped
> to the actual NSE trading calendar so Indian market holidays are
> excluded.

------------------------------------------------------------------------

## 📁 Project Structure

A recommended project structure is:

``` text
IOC-Stock-Forecasting/
│
├── IOC_Stock_Price_Forecasting_ARIMA_SARIMA.ipynb
├── IOC_NS_stock_data.csv
├── README.md

```

The CSV file is generated by the notebook during the data-collection
stage.

------------------------------------------------------------------------

## ⚙️ Installation

Install the required Python libraries:

``` bash
pip install numpy pandas matplotlib seaborn yfinance statsmodels scikit-learn
```



------------------------------------------------------------------------

## ▶️ How to Run

1.  Clone or download the project.
2.  Install the required Python packages.
3.  Open:

``` text
IOC_Stock_Price_Forecasting_ARIMA_SARIMA.ipynb
```

4.  Run the notebook cells sequentially from top to bottom.
5.  The notebook downloads the latest data available within the
    specified date range.
6.  The cleaned dataset is saved as:

``` text
IOC_NS_stock_data.csv
```

7.  The notebook performs the complete analysis and produces the final
    forecasts and diagnostic plots.

### Internet requirement

Because the notebook uses `yfinance` to download historical market data,
an internet connection is required when running the data-collection
section.

------------------------------------------------------------------------

## 🧰 Technologies Used

-   **Python**
-   **NumPy**
-   **Pandas**
-   **Matplotlib**
-   **Seaborn**
-   **yfinance**
-   **Statsmodels**
-   **Scikit-learn**

### Main statistical techniques

-   Exploratory Data Analysis
-   Augmented Dickey-Fuller Test
-   Differencing
-   Log Returns
-   ACF/PACF
-   Seasonal Decomposition
-   ARIMA
-   SARIMA
-   Walk-forward Validation
-   Naïve Random-Walk Benchmark
-   MAE
-   RMSE
-   Ljung--Box Test
-   ARCH Test
-   Q-Q Plot
-   Prediction Intervals

------------------------------------------------------------------------

## 💡 Key Findings

1.  The original IOC closing-price series is non-stationary.
2.  First-order differencing produces a stationary series according to
    the ADF test.
3.  ARIMA(1,1,1) is the best ARIMA model according to validation RMSE.
4.  SARIMA(1,1,1)(1,0,1,5) achieves a slightly lower validation RMSE
    than ARIMA.
5.  ARIMA and SARIMA provide only marginal improvements over the naïve
    random-walk benchmark.
6.  Residual diagnostics indicate remaining dependence at some longer
    lags.
7.  The ARCH test provides strong evidence of time-varying residual
    variance.
8.  The residual Q-Q plot indicates departures from normality,
    particularly in the tails.
9.  The selected SARIMA model is used to produce a 30-step future
    forecast with 95% confidence intervals.
10. The results suggest that daily IOC closing prices have limited
    predictable linear structure using the ARIMA-family models
    considered.

------------------------------------------------------------------------

## ⚠️ Limitations

This project has several limitations:

-   Only IOC.NS is analysed.
-   The modelling focuses on closing prices rather than building a
    dedicated return/volatility forecasting framework.
-   The SARIMA seasonal period of 5 is an exploratory representation of
    a trading week; it does not establish strong weekly seasonality.
-   The selected SARIMA model has no seasonal AR, differencing, or MA
    terms: `(1,0,1,5)`.
-   Residual diagnostics indicate heteroskedasticity, but a GARCH model
    is not included.
-   The models do not incorporate external variables such as crude-oil
    prices, market indices, interest rates, macroeconomic variables, or
    company-specific information.
-   Future dates are currently generated using weekday business dates
    rather than an NSE-specific holiday calendar.
-   ARIMA-family models mainly capture linear temporal dependence and
    may not capture nonlinear market dynamics.

------------------------------------------------------------------------

## 🚀 Future Improvements

Possible extensions include:

### 1. GARCH

Use an ARIMA/GARCH framework to model the changing volatility identified
by the ARCH test.

### 2. KPSS Test

Use the KPSS test alongside ADF to provide a stronger assessment of
stationarity.

### 3. NSE Trading Calendar

Replace generic business-day dates with actual NSE trading sessions when
creating future forecast dates.

### 4. Return-Based Modelling

Investigate forecasting log returns and volatility separately from
price-level forecasting.

### 5. Machine Learning Models

Compare the statistical models with models such as:

-   Random Forest
-   XGBoost

using lagged prices, returns, rolling statistics, and other features.

### 6. Additional Statistical Models

Consider:

-   Exponential Smoothing
-   ETS
-   GARCH
-   Other volatility models

### 7. External Predictors

Investigate whether IOC forecasting performance improves when
incorporating relevant external variables.

------------------------------------------------------------------------

## 📌 Important Interpretation

This project should **not** be interpreted as demonstrating that ARIMA
or SARIMA can reliably predict future stock prices.

The main statistical finding is that the sophisticated ARIMA-family
models produce only a small improvement over a simple naïve random-walk
benchmark.

Therefore, the project is better interpreted as an investigation of the
**predictable time-series structure and forecasting limitations of IOC
daily closing prices**, rather than as a claim of highly accurate
stock-price prediction.

------------------------------------------------------------------------

## 👨‍💻 Project

**Title:** Forecasting Indian Oil Corporation Stock Prices Using ARIMA
and SARIMA: A Time Series Approach

**Notebook:** `IOC_Stock_Price_Forecasting_ARIMA_SARIMA.ipynb`

**Asset:** Indian Oil Corporation --- `IOC.NS`
