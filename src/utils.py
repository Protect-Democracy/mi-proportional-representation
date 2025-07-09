import geopandas as gp


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
    result = large.merge(
        aggregated_data,
        left_on=identifier_column,
        right_index=True,  # Merge on the index of the aggregated data
        how="left",
    )

    # Replace any potential NaN values with 0 for cleaner data
    result[numeric_cols_to_sum] = result[numeric_cols_to_sum].fillna(0)

    return result


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
