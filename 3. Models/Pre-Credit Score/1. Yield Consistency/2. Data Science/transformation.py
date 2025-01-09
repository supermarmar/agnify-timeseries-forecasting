import numpy as np
import pandas as pd


def clean_delivery_data(
    df: pd.DataFrame, target_variable: str, crop: str
) -> pd.DataFrame:
    df[target_variable] = np.where(df[target_variable] < 0, 0, df[target_variable])

    df = df[df["Crop"] == crop].drop(columns=["Crop"]).reset_index().set_index("Date")

    return df


def clean_production_data(
    df: pd.DataFrame, target_variable: str, crop: str, year: int
) -> pd.DataFrame:
    df[target_variable] = np.where(df[target_variable] < 0, 0, df[target_variable])

    df = df[df["Crop"] == crop].drop(columns=["Crop"]).reset_index().set_index("Date")

    df = df[df.index.year <= year]

    return df


def resample_data(df: pd.DataFrame, target_variable: str) -> pd.DataFrame:
    series_df = df[target_variable]
    # Resample the data to weekly frequency (aggregating or filling in missing values)
    resampled_df = series_df.resample("W").ffill().to_frame()
    return resampled_df


def smooth_series(
    resampled_df: pd.DataFrame,
    target_variable: str,
    rolling_window: int,
    smoothed_target_variable: str,
) -> pd.Series:
    # Calculating a rolling average
    smoothed_df = resampled_df.copy()
    smoothed_df[smoothed_target_variable] = (
        smoothed_df[target_variable].rolling(window=rolling_window).mean().fillna(0)
    )
    smoothed_series = smoothed_df[smoothed_target_variable]
    return smoothed_series
