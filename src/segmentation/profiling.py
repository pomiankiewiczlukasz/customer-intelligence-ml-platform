from __future__ import annotations

import pandas as pd


def profile_segments(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a business-oriented profile for each customer segment.
    """

    profile = (
        df.groupby("segment")
        .agg(
            customers=("segment", "size"),
            churn_rate=(
                "Churn",
                lambda x: (x == "Yes").mean(),
            ),
            avg_tenure=("tenure", "mean"),
            avg_monthly_charges=("MonthlyCharges", "mean"),
            avg_total_charges=("TotalCharges", "mean"),
        )
        .reset_index()
    )

    return profile