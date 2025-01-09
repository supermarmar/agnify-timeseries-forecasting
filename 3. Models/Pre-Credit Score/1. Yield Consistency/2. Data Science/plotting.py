import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.model_selection import TimeSeriesSplit

fig_size = (14, 7)


def plot_anomalies(
    outlier_df: pd.DataFrame,
    anomalies: pd.DataFrame,
    target_variable: str,
    figsize: tuple = fig_size,
) -> None:
    plt.figure(figsize=figsize)

    # Plot the TARGET_VARIABLE as a black line
    sns.lineplot(
        data=outlier_df,
        x=outlier_df.index,
        y=target_variable,
        color="black",
    )

    # Plot the anomaly points as red dots
    sns.scatterplot(
        data=anomalies,
        x=anomalies.index,
        y=target_variable,
        color="red",
        label="Anomalies",
    )


def plot_anomalies_and_rolling(
    outlier_df: pd.DataFrame,
    smoothed_series: pd.Series,
    anomalies: pd.DataFrame,
    target_variable: str,
    smoothed_target_variable: str,
    figsize: tuple = fig_size,
) -> None:
    plt.figure(figsize=figsize)

    # Plot the TARGET_VARIABLE as a black line
    sns.lineplot(
        data=outlier_df,
        x=outlier_df.index,
        y=target_variable,
        color="black",
        label=target_variable,
    )

    # Plot the SMOOTHED_TARGET_VARIABLE as an orange line
    sns.lineplot(
        data=smoothed_series.to_frame(),
        x=smoothed_series.index,
        y=smoothed_target_variable,
        color="orange",
        label=smoothed_target_variable,
    )

    # Plot the anomaly points as red dots
    sns.scatterplot(
        data=anomalies,
        x=anomalies.index,
        y=target_variable,
        color="red",
        label="Anomalies",
    )


def plot_forecast(
    forecast: pd.Series,
    smoothed_series: pd.Series,
    cleaned_df: pd.DataFrame,
    target_variable: str,
    smoothed_target_variable: str,
    period: int,
    num_years: int,
    figsize: tuple = fig_size,
) -> None:

    smooth_df = smoothed_series.to_frame()
    forecast_df = forecast.to_frame()

    # Create a date range for the forecast
    forecast_dates = pd.date_range(
        start=smoothed_series.index[-1] + pd.Timedelta(days=7),
        periods=period * num_years,
        freq="W",
    )
    forecast_series = pd.Series(forecast, index=forecast_dates)

    # Plotting the forecast
    plt.figure(figsize=figsize)
    sns.lineplot(
        data=cleaned_df, x=cleaned_df.index, y=target_variable, label="Actuals"
    )
    sns.lineplot(
        data=smooth_df,
        x=smoothed_series.index,
        y=smooth_df[smoothed_target_variable],
        label="Smoothed Series",
    )
    sns.lineplot(
        data=forecast_df,
        x=forecast_df.index,
        y=forecast_df["predicted_mean"],
        color="red",
        label="Forecast",
    )


def plot_train_splits(
    tscv: TimeSeriesSplit,
    series: pd.Series,
    figsize: tuple = fig_size,
) -> None:
    plt.figure(figsize=figsize)

    for i, (train_index, test_index) in enumerate(tscv.split(series)):
        train_data = series.iloc[train_index]

        # Plot train splits with transparency
        plt.plot(train_data.index, train_data, label=f"Train Split {i+1}", alpha=0.5)

        # Get the train date range
        train_start, train_end = train_data.index[0], train_data.index[-1]

        # Print train date range and number of samples
        print(
            f"Train Split {i+1}: {train_start.date()} to {train_end.date()} ({len(train_data)} samples)"
        )

    plt.title("TimeSeriesSplit - Train Splits")
    plt.show()


def plot_test_splits(
    tscv: TimeSeriesSplit,
    series: pd.Series,
    figsize: tuple = fig_size,
) -> None:
    plt.figure(figsize=figsize)

    for i, (train_index, test_index) in enumerate(tscv.split(series)):
        test_data = series.iloc[test_index]

        # Plot test splits with transparency
        plt.plot(
            test_data.index,
            test_data,
            label=f"Test Split {i+1}",
            linestyle="--",
            alpha=0.7,
        )

        # Get the test date range
        test_start, test_end = test_data.index[0], test_data.index[-1]

        # Print test date range and number of samples
        print(
            f"Test Split {i+1}: {test_start.date()} to {test_end.date()} ({len(test_data)} samples)"
        )

    plt.title("TimeSeriesSplit - Test Splits")
    plt.show()


def plot_eighty_twenty_series(
    train: pd.Series,
    test: pd.Series,
    figsize: tuple = fig_size,
) -> None:
    plt.figure(figsize=figsize)

    # Plot the training series
    sns.lineplot(x=train.index, y=train, label="Training Series")

    # Plot the testing series
    sns.lineplot(x=test.index, y=test, label="Testing Series")
