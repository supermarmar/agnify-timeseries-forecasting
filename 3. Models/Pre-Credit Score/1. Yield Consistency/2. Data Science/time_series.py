import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
import statsmodels.api as sm
from prophet import Prophet


def find_best_arima_order(
    series: pd.Series, max_p: int = 2, max_q: int = 2, d: int = 0
) -> tuple:
    best_aic = float("inf")
    best_order = None

    for p in range(max_p + 1):
        for q in range(max_q + 1):
            try:
                model = sm.tsa.ARIMA(series, order=(p, d, q))
                results = model.fit()
                if results.aic < best_aic:
                    best_aic = results.aic
                    best_order = (p, d, q)
            except:
                continue

    return best_order, best_aic


def find_best_sarima_model(series: pd.Series, period: int) -> tuple:
    best_aic = float("inf")
    best_order = None
    best_seasonal_order = None
    d = 0
    D = 1

    for p in range(3):  # Test p = 0, 1, 2
        for q in range(3):  # Test q = 0, 1, 2
            for P in range(3):  # Test P = 0, 1, 2
                for Q in range(3):  # Test Q = 0, 1, 2
                    try:
                        model = sm.tsa.SARIMAX(
                            series, order=(p, d, q), seasonal_order=(P, D, Q, period)
                        )
                        result = model.fit(disp=False)
                        if result.aic < best_aic:
                            best_aic = result.aic
                            best_order = (p, d, q)
                            best_seasonal_order = (P, D, Q, period)
                    except:
                        continue

    return best_order, best_seasonal_order


def evaluate_models(
    data: pd.Series, tscv, period: int, model_parms: dict
) -> pd.DataFrame:
    results = []

    for train_idx, test_idx in tscv.split(data):
        train, test = data.iloc[train_idx], data.iloc[test_idx]

        # Exponential Smoothing
        exp_model = sm.tsa.ExponentialSmoothing(
            train, trend="add", seasonal="add", seasonal_periods=period
        )
        exp_fit = exp_model.fit()
        exp_forecast = exp_fit.forecast(len(test))

        # Prophet
        prophet_train = pd.DataFrame({"ds": train.index, "y": train.values})
        prophet_model = Prophet()
        prophet_model.fit(prophet_train)
        future = pd.DataFrame({"ds": test.index})
        prophet_forecast = prophet_model.predict(future)["yhat"]

        # SARIMA
        sarima_model = sm.tsa.SARIMAX(
            train,
            order=model_parms["order"],
            seasonal_order=model_parms["seasonal_order"],
        )
        sarima_fit = sarima_model.fit(disp=False)
        sarima_forecast = sarima_fit.forecast(steps=len(test))

        # Evaluate Forecasts
        metrics = {
            "split": len(results) + 1,
            "sarima_mae": mean_absolute_error(test, sarima_forecast),
            "sarima_rmse": np.sqrt(mean_squared_error(test, sarima_forecast)),
            "exp_mae": mean_absolute_error(test, exp_forecast),
            "exp_rmse": np.sqrt(mean_squared_error(test, exp_forecast)),
            "prophet_mae": mean_absolute_error(test, prophet_forecast),
            "prophet_rmse": np.sqrt(mean_squared_error(test, prophet_forecast)),
        }
        results.append(metrics)

    return pd.DataFrame(results)


def rank_models(results: pd.DataFrame) -> pd.DataFrame:
    avg_results = results.mean()

    # Create a dataframe with the models as index and RMSE and MAE scores as columns
    ranked_results_df = pd.DataFrame(
        {
            "Model": ["SARIMA", "Exponential Smoothing", "Prophet"],
            "MAE": [
                avg_results["sarima_mae"],
                avg_results["exp_mae"],
                avg_results["prophet_mae"],
            ],
            "RMSE": [
                avg_results["sarima_rmse"],
                avg_results["exp_rmse"],
                avg_results["prophet_rmse"],
            ],
        }
    )

    # Rank the models based on MAE and RMSE
    ranked_results_df["MAE Rank"] = ranked_results_df["MAE"].rank()
    ranked_results_df["RMSE Rank"] = ranked_results_df["RMSE"].rank()

    return ranked_results_df


def mae_rmse_mape_calc(test: pd.Series, forecast: pd.Series) -> tuple:
    mae = mean_absolute_error(test, forecast)
    rmse = np.sqrt(mean_squared_error(test, forecast))
    mape = np.mean(np.abs((test - forecast) / test)) * 100
    print(f"SARIMA - MAE: {mae}, RMSE: {rmse}, MAPE: {mape}")
    return mae, rmse


def fit_sarima_model(train: pd.Series, test: pd.Series, model_params: dict) -> tuple:
    # Fit SARIMA model
    sarima_model = sm.tsa.SARIMAX(
        train,
        order=model_params["order"],
        seasonal_order=model_params["seasonal_order"],
    )
    sarima = sarima_model.fit()

    # Forecasting for the test period
    sarima_forecast = sarima.forecast(steps=len(test))

    mae_sarima, rmse_sarima = mae_rmse_mape_calc(test, sarima_forecast)

    return sarima, sarima_forecast, mae_sarima, rmse_sarima


def fit_exp_smooth_model(train: pd.Series, test: pd.Series, period: int) -> tuple:
    # Fit the model
    exp_smooth_model = sm.tsa.ExponentialSmoothing(
        train, trend="add", seasonal="add", seasonal_periods=period
    )
    exp_smooth = exp_smooth_model.fit()

    # Forecasting for the test period
    exp_smooth_forecast = exp_smooth.forecast(steps=len(test))

    mae_exp_smooth, rmse_exp_smooth = mae_rmse_mape_calc(test, exp_smooth_forecast)

    return exp_smooth, exp_smooth_forecast, mae_exp_smooth, rmse_exp_smooth


def fit_prophet_model(fb_df: pd.DataFrame, test: pd.Series) -> tuple:
    # Fit the model
    model = Prophet()
    model.fit(fb_df)

    # Forecast for the next 52 weeks (for example)
    future = model.make_future_dataframe(periods=len(test), freq="W")  # weekly forecast
    forecast = model.predict(future)

    # Extract the forecasted values
    prophet_forecast = forecast.set_index("ds").loc[test.index]["yhat"]

    (
        mae_prophet,
        rmse_prophet,
    ) = mae_rmse_mape_calc(test, prophet_forecast)

    return model, prophet_forecast, mae_prophet, rmse_prophet


def display_metrics(metrics_df: pd.DataFrame) -> pd.DataFrame:
    # Sort the DataFrame by MAE and then by RMSE
    metrics_df_sorted = metrics_df.sort_values(by=["MAE", "RMSE"])

    # Add ranking columns
    metrics_df_sorted["MAE Rank"] = metrics_df_sorted["MAE"].rank()
    metrics_df_sorted["RMSE Rank"] = metrics_df_sorted["RMSE"].rank()

    # Display the sorted DataFrame with rankings
    return metrics_df_sorted
