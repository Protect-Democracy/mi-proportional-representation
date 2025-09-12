import geopandas as gp
from typing import Any, Dict
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm


def dissolve_small_into_large(small, large, identifier_column):
    """
    Aggregates numeric data from small geometries into larger ones by
    summing the values based on a spatial join.

    Args:
        small (gpd.GeoDataFrame): A GeoDataFrame with smaller geometries
                                  (e.g., census blocks).
        large (gpd.GeoDataFrame): A GeoDataFrame with larger geometries
                                  (e.g., voting precincts).
        identifier_column (str): The column name in the 'large' dataframe
                                 that serves as a unique identifier.

    Returns:
        gpd.GeoDataFrame: The 'large' GeoDataFrame with new columns containing
                          the summed numeric data from 'small'.
    """
    # Use representative points for a stable spatial join
    small_with_points = small.copy()
    small_with_points["geometry"] = small.geometry.representative_point()

    # Spatially join the small geometries (as points) to the large ones
    joined = gp.sjoin(small_with_points, large, how="inner", predicate="within")

    # IMPORTANT: Identify the numeric columns to sum *only* from the small dataframe
    numeric_cols_to_sum = small.select_dtypes(include="number").columns.tolist()

    # Group by the large geometry identifier and sum only the specified numeric columns
    aggregated_data = joined.groupby(identifier_column)[numeric_cols_to_sum].sum()

    # Merge the aggregated data back into the large dataframe.
    # A 'left' merge keeps all geometries from the 'large' dataframe.
    return large.merge(
        aggregated_data,
        left_on=identifier_column,
        right_index=True,  # Merge on the index of the aggregated data
        how="left",
    )

    # Replace any potential NaN values with 0 for cleaner data


def fit_statistical_models(data: pd.DataFrame) -> Dict[str, Any]:
    """
    Fits statistical models for ticket-splitting (crossover voting).

    Args:
        data: The precinct-level data containing voting results.

    Returns:
        A dictionary containing the fitted models.
    """
    print("Fitting statistical models for vote simulation...")

    # Fit crossover voting model (ticket-splitting)
    vote_diff = data["DONALD J. TRUMP"] / (
        data["KAMALA D. HARRIS"] + data["DONALD J. TRUMP"]
    ) - data["STATE_REP_GOP"] / (data["STATE_REP_DEM"] + data["STATE_REP_GOP"])
    hist, bins = np.histogram(vote_diff, bins=np.linspace(-0.5, 0.5, 201), density=True)

    def norm_loss(params, x=bins[:-1], y=hist):
        return np.mean(((norm.pdf(x, loc=params[0], scale=params[1])) - y) ** 2)

    crossover_res = minimize(
        norm_loss, x0=[0, 0.02], tol=1e-4, options={"maxiter": 100}
    )

    return {"crossover": crossover_res}


def modify_2party_share(value: float, alpha: float) -> float:
    """Applies a partisan trend shift to the 2-party vote share."""
    if alpha > 0:  # Shift towards GOP
        return value - alpha * (value - value**2)
    if alpha < 0:  # Shift towards DEM
        return value + alpha * (value - value**0.5)
    return value


def label_small_with_large(small, large, identifier_column, label_column_name=None):
    if label_column_name is None:
        label_column_name = identifier_column

    # Use representative points for stable point-in-polygon join
    small_points = small.copy()
    small_points["geometry"] = small.geometry.representative_point()

    # Perform spatial join: each point gets the identifier of the large geometry it falls within
    joined = gp.sjoin(
        small_points,
        large[[identifier_column, "geometry"]],
        how="left",
        predicate="within",
    ).drop_duplicates()

    # Extract the labeling column and align with the original index
    labels = joined[[identifier_column]]
    labels.columns = [label_column_name]

    # Merge the labels back into the original 'small' dataframe
    result = small.copy()
    result[label_column_name] = labels[label_column_name]

    return result
