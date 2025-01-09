import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from typing import Tuple


def detect_anomalies(
    df: pd.DataFrame, target_variable: str, outliers_fraction: float = 0.2
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    scaler = StandardScaler()
    np_scaled = scaler.fit_transform(df[target_variable].values.reshape(-1, 1))
    scaled_df = pd.DataFrame(np_scaled)

    model = IsolationForest(contamination=outliers_fraction)
    model.fit(scaled_df)

    outlier_df = df.copy()
    outlier_df["Anomaly"] = model.predict(scaled_df)
    anomalies = outlier_df.loc[outlier_df["Anomaly"] == -1]

    return outlier_df, anomalies


def calculate_anomaly_percentage(outlier_df: pd.DataFrame) -> pd.DataFrame:
    # Calculate the percentage of anomalies and non-anomalies
    anomaly_counts = outlier_df["Anomaly"].value_counts(normalize=True) * 100

    # Create a new dataframe to display the results
    anomaly_df = pd.DataFrame(
        {
            "Anomaly": ["Not Anomaly", "Anomaly"],
            "Percentage": [anomaly_counts[1], anomaly_counts[-1]],
        }
    )
    return anomaly_df
