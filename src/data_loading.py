import geopandas as gp
import pandas as pd
import numpy as np
import requests
import zipfile
import os

from src.utils import dissolve_small_into_large, label_small_with_large


def load_and_format_votes():
    names_cols = [
        "ElectionYear",
        "ElectionType",
        "OfficeCode",
        "DistrictCode",
        "StatusCode",
        "CandidateID",
        "LastName",
        "FirstName",
        "MiddleName",
        "Party",
    ]
    votes_cols = [
        "ElectionYear",
        "ElectionType",
        "OfficeCode",
        "DistrictCode",
        "StatusCode",
        "CandidateID",
        "CountyCode",
        "CityTownCode",
        "WardNumber",
        "PrecinctNumber",
        "PrecinctLabel",
        "Votes",
    ]
    cities_cols = [
        "ElectionYear",
        "ElectionType",
        "CountyCode",
        "CityTownCode",
        "CityTownName",
    ]

    # Load data from files
    votes = pd.read_csv(
        "raw_data/2024GEN/2024vote.txt",
        delimiter="\t",
        names=votes_cols,
        low_memory=False,
    )

    names = pd.read_csv(
        "raw_data/2024GEN/2024name.txt",
        delimiter="\t",
        names=names_cols,
        low_memory=False,
    )

    cities = pd.read_csv(
        "raw_data/2024GEN/2024city.txt",
        delimiter="\t",
        names=cities_cols,
        low_memory=False,
    )

    # Merge datasets
    votes["CandidateID"] = votes["CandidateID"].astype(str)
    names["CandidateID"] = names["CandidateID"].astype(str)
    name_fields = ["CandidateID", "FirstName", "MiddleName", "LastName", "Party"]
    names_subset = names[name_fields]
    # merge candidate names into votes
    votes = votes.merge(names_subset, on="CandidateID", how="left")

    cities_subset = cities[["CityTownName", "CityTownCode"]]

    # merge city names into votes
    votes = votes.merge(cities_subset, on="CityTownCode", how="left")

    votes["CityTownName"] = votes["CityTownName"].apply(
        lambda x: x.replace("CHARTER ", "").replace("TWP", "TOWNSHIP")
    )
    votes["CityTownName"] = votes["CityTownName"].str.replace(
        r"^(TOWNSHIP|CITY|VILLAGE) OF (.+)",
        lambda m: f"{m.group(2)} {m.group(1)}",
        regex=True,
    )

    # Format full name
    votes["Candidate Name"] = (
        votes[["FirstName", "MiddleName", "LastName"]]
        .fillna("")
        .agg(lambda x: " ".join(part for part in x if part.strip()), axis=1)
    )

    # Combine precinct # and letter into a single field to match the shapefile
    votes["PrecinctNumber"] = votes["PrecinctNumber"].astype(str) + votes[
        "PrecinctLabel"
    ].fillna("")

    # Remove "statistical adjustments" which are only a few hundred votes across the state
    votes = votes[votes["PrecinctNumber"] != "9999"]

    votes = votes.drop_duplicates()  # this now matches the official tallies

    # Strip out non-POTUS and non state rep candidates
    votes = votes[votes["OfficeCode"].apply(lambda x: x == 1 or x == 8)]

    # Pivot so now we have 1 row per precinct, 1 column per candidate
    votes = votes.pivot(
        index=["CountyCode", "PrecinctNumber", "WardNumber", "CityTownName"],
        columns="Candidate Name",
        values="Votes",
    ).reset_index()

    # Make new columns for state representatives (republican/democrat/other)
    # Bespoke fixes
    individual_fixes = {
        "MANCHESTER CITY": "MANCHESTER TOWNSHIP",
    }
    votes["CityTownName"] = votes["CityTownName"].apply(
        lambda x: individual_fixes[x] if x in individual_fixes.keys() else x
    )
    votes = votes[[v for v in votes.columns if v != ""]]

    # Extract state representative results & consolidate
    names["Candidate Name"] = (
        names[["FirstName", "MiddleName", "LastName"]]
        .fillna("")
        .agg(lambda x: " ".join(part for part in x if part.strip()), axis=1)
    )

    state_dem_candidates = names[
        (names["OfficeCode"] == 8) & (names["Party"] == "DEM")
    ]["Candidate Name"]
    cols_to_sum = [col for col in votes.columns if col in state_dem_candidates.values]
    # Sum the values across these columns for every row, all at once
    votes["STATE_REP_DEM"] = votes[cols_to_sum].sum(axis=1)

    state_gop_candidates = names[
        (names["OfficeCode"] == 8) & (names["Party"] == "REP")
    ]["Candidate Name"]
    cols_to_sum = [col for col in votes.columns if col in state_gop_candidates.values]
    # Sum the values across these columns for every row, all at once
    votes["STATE_REP_GOP"] = votes[cols_to_sum].sum(axis=1)

    candidate_columns = [
        "KAMALA D. HARRIS",
        "DONALD J. TRUMP",
        "STATE_REP_GOP",
        "STATE_REP_DEM",
    ]

    return votes[
        [
            c
            for c in candidate_columns
            + ["CountyCode", "PrecinctNumber", "WardNumber", "CityTownName"]
        ]
    ], candidate_columns


def reallocate_detroit_counting_board_votes(votes, candidate_columns):
    crosswalk = pd.read_csv("raw_data/detroit_crosswalk.csv")

    # Split up Detroit & non-Detroit data
    is_detroit = votes["CityTownName"].str.contains("detroit", case=False)
    non_detroit_votes = votes[~is_detroit].copy()
    detroit_votes = votes[is_detroit].copy()

    # Separate counting board (CB) precincts from geographical precincts in Detroit
    is_cb = detroit_votes["PrecinctNumber"].str.contains("CB", na=False)
    cb_votes = detroit_votes[is_cb].copy().fillna(0.0)
    geo_precincts = detroit_votes[~is_cb].copy()

    # Clean up precinct numbers for merging
    cb_votes["cbprecinct"] = (
        cb_votes["PrecinctNumber"].str.replace("CB", "").astype(str)
    )
    cb_votes["cbprecinct"] = cb_votes["cbprecinct"].apply(lambda x: int(str(x)[1:]))
    geo_precincts["PrecinctNumber"] = geo_precincts["PrecinctNumber"].astype(int)

    # Merge geo precincts with the crosswalk to get their CB assignment
    merged_detroit = pd.merge(
        geo_precincts,
        crosswalk,
        left_on="PrecinctNumber",
        right_on="precinct",
        how="left",
    ).fillna(0)

    sub_geos = []
    for cbprecinct in cb_votes["cbprecinct"].unique():
        subcb = cb_votes[cb_votes["cbprecinct"] == cbprecinct]
        sub_geo = merged_detroit[merged_detroit["cbprecinct"] == cbprecinct]
        # Calculate the sum of in-person votes for each CB group
        group_proportions = (
            (sub_geo[candidate_columns] / sub_geo[candidate_columns].sum())
            .astype(float)
            .fillna(0.0)
        )
        group_totals = sub_geo[
            candidate_columns
        ].values + group_proportions.values * subcb[candidate_columns].values.reshape(
            1, -1
        ).repeat(repeats=len(subcb), axis=0)
        group_totals = group_totals.astype(int)
        sub_geo.loc[:, candidate_columns] = group_totals
        sub_geos.append(sub_geo)

    new_detroit_votes = pd.concat(sub_geos).fillna(0.0)

    # Drop the helper columns from the merge
    new_detroit_votes.drop(columns=["precinct", "cbprecinct"], inplace=True)

    # Combine the updated Detroit data with the rest of the state's data
    new_votes = pd.concat([non_detroit_votes, new_detroit_votes], ignore_index=True)
    new_votes["PrecinctNumber"] = new_votes["PrecinctNumber"].astype(str)
    return new_votes


def load_and_format_precincts_shapefile():
    # Now look at shapefile, which has 4339 precincts
    df = gp.read_file("raw_data/2024_Voting_Precincts/2024_Voting_Precincts.shp")

    # Fix a typo
    df.loc[df["Precinct_L"] == "3Macomb Township, Precinct 23", "Precinct_L"] = (
        "Macomb Township, Precinct 23"
    )

    df["COUNTYFIPS"] = df["COUNTYFIPS"].apply(lambda x: int(str(x)))  # turn into ints
    df["CountyCode"] = (
        df["COUNTYFIPS"].rank(method="dense").astype(int)
    )  # match county codes in vote data

    df["PrecinctNumber"] = df["Precinct_L"].apply(
        lambda x: str(x).split(" ")[-1]
    )  # get precinct from description

    df["CityTownName"] = df["Precinct_L"].apply(lambda x: str(x).split(",")[0].upper())

    # Apply regex replacement
    df["CityTownName"] = df["CityTownName"].astype(str).str.strip()

    df["CityTownName"] = df["CityTownName"].apply(
        lambda x: x.replace("CHARTER ", "").replace("TWP", "TOWNSHIP")
    )
    df["CityTownName"] = df["CityTownName"].str.replace(
        r"^(TOWNSHIP|CITY|VILLAGE) OF (.+)",
        lambda m: f"{m.group(2)} {m.group(1)}",
        regex=True,
    )
    df["WardNumber"] = df["WARD"].astype(int)

    # Various fixes for edge cases
    # Flint Township, precinct 2 has its election year as 2023
    df["ELECTIONYE"] == 2024
    # Grand blanc has a typo
    df.loc[
        np.logical_and(
            df["CityTownName"].str.contains("GRAND BLANC"), df["PrecinctNumber"] == "12"
        ),
        "PrecinctNumber",
    ] = "10"

    df.loc[df["CityTownName"] == "NONE", "PrecinctNumber"] = "1W"
    individual_fixes = {
        "L'ANSE TOWNSHIP": "LANSE TOWNSHIP",
        "THE VILLAGE OF CLARKSTON CITY": "CLARKSTON CITY VILLAGE",
        "THE VILLAGE OF DOUGLAS CITY": "DOUGLAS CITY VILLAGE",
        "GROSSE POINTE SHORES VILLAGE": "GROSSE POINTE SHORES CITY",
        "SAULT STE. MARIE CITY": "SAULT STE MARIE CITY",
        "ST. CHARLES TOWNSHIP": "ST CHARLES TOWNSHIP",
        "ST. CLAIR CITY": "ST CLAIR CITY",
        "ST. CLAIR SHORES CITY": "ST CLAIR SHORES CITY",
        "ST. CLAIR TOWNSHIP": "ST CLAIR TOWNSHIP",
        "ST. IGNACE CITY": "ST IGNACE CITY",
        "ST. IGNACE TOWNSHIP": "ST IGNACE TOWNSHIP",
        "ST. JAMES TOWNSHIP": "ST JAMES TOWNSHIP",
        "ST. JOHNS CITY": "ST JOHNS CITY",
        "ST. JOSEPH CITY": "ST JOSEPH CITY",
        "ST. JOSEPH TOWNSHIP": "ST JOSEPH TOWNSHIP",
        "ST. LOUIS CITY": "ST LOUIS CITY",
        "NONE": "MILAN CITY",
    }
    df["CityTownName"] = df["CityTownName"].apply(
        lambda x: individual_fixes[x] if x in individual_fixes.keys() else x
    )

    # fix jackson - this seems like a fat finger as well
    df.loc[
        np.logical_and(
            np.logical_and(
                df["CityTownName"].str.contains("JACKSON"),
                df["PrecinctNumber"] == "6",
            ),
            df["WARD"] == "06",
        ),
        "PrecinctNumber",
    ] = "9"

    return df


def merge_shapefiles_and_votes(votes, shapefiles):
    # perform the big merge
    merged = votes.merge(
        shapefiles,
        on=["CityTownName", "WardNumber", "CountyCode", "PrecinctNumber"],
        how="outer",
        indicator=True,
    )
    # 3818 precincts merge perfectly on name/county/number/ward
    inner = merged[merged["_merge"] == "both"]

    left = merged[merged["_merge"] == "left_only"]
    right = merged[merged["_merge"] == "right_only"]
    left.drop("geometry", inplace=True, axis=1)
    # Most of the rest of them merge if we ignore ward
    noward = left[[c for c in votes.columns]].merge(
        right[[c for c in shapefiles.columns]],
        on=["CityTownName", "CountyCode", "PrecinctNumber"],
        how="inner",
    )

    combined = pd.concat([inner, noward], axis=0).drop_duplicates()

    combined.drop("_merge", axis=1, inplace=True)
    return gp.GeoDataFrame(combined).set_crs("EPSG:3078")


def load_legislature(precincts):
    house2024 = gp.read_file("raw_data/tiger_lower_2024/tl_2024_26_sldl.shp")
    house2024 = house2024.to_crs("EPSG:3078")

    senate2024 = gp.read_file("raw_data/tiger_upper_2024/tl_2024_26_sldu.shp")
    senate2024 = senate2024.to_crs("EPSG:3078")

    house2024 = dissolve_small_into_large(
        precincts, house2024, identifier_column="SLDLST"
    )
    senate2024 = dissolve_small_into_large(
        precincts, senate2024, identifier_column="SLDUST"
    )

    house2024["2partyshare"] = house2024["DONALD J. TRUMP"] / (
        house2024["DONALD J. TRUMP"] + house2024["KAMALA D. HARRIS"]
    )

    senate2024["2partyshare"] = senate2024["DONALD J. TRUMP"] / (
        senate2024["DONALD J. TRUMP"] + senate2024["KAMALA D. HARRIS"]
    )

    # Legislator data
    leg = pd.read_stata("raw_data/legislator_data/shor_mccarty.dta")
    leg = leg[leg["st"] == "MI"]
    # put names in first->last name order
    leg["name"] = leg["name"].apply(lambda x: " ".join(x.split(", ")[::-1]))
    house2024 = house2024.merge(
        leg, left_on="SLDLST", right_on="hdistrict2022", how="left"
    )
    senate2024 = senate2024.merge(
        leg, left_on="SLDUST", right_on="sdistrict2022", how="left"
    )

    house2024.drop_duplicates("geometry", inplace=True)
    senate2024.drop_duplicates("geometry", inplace=True)
    # current legislature makeup
    plural = pd.read_csv("raw_data/plural.csv")

    # Let's see how many people are in both the shor-mccarty data
    # and were also elected in 2024
    house_subset = house2024[
        [
            n in list(set(plural.name).intersection(set(leg.name)))
            for n in house2024.name
        ]
    ]

    senate_subset = senate2024[
        [
            n in list(set(plural.name).intersection(set(leg.name)))
            for n in senate2024.name
        ]
    ]

    return house2024, senate2024, house_subset, senate_subset


def load_tiger_blocks():
    """
    Loads the TIGER/Line block shapefile for Michigan (FIPS 26) for the year 2024.

    This function checks for a local copy of the shapefile first. If it's not
    found, it downloads the zip archive from the US Census Bureau's FTP server,
    extracts it, and then loads it into a GeoDataFrame. The data is then
    re-projected to the specified CRS ("EPSG:3078").

    Returns:
        geopandas.GeoDataFrame: A GeoDataFrame containing the census blocks.
    """
    # Define the URL for the TIGER shapefile and the local file paths
    url = "https://www2.census.gov/geo/tiger/TIGER2024/TABBLOCK20/tl_2024_26_tabblock20.zip"
    local_dir = "raw_data/tiger_blocks"
    shp_filename = "tl_2024_26_tabblock20.shp"
    local_shp_path = os.path.join(local_dir, shp_filename)
    local_zip_path = os.path.join(local_dir, os.path.basename(url))

    # Check if the shapefile already exists. If not, download and extract it.
    if not os.path.exists(local_shp_path):
        print(f"Shapefile not found locally. Downloading from {url}...")

        # Create the local directory if it doesn't exist
        os.makedirs(local_dir, exist_ok=True)

        # Download the file
        try:
            with requests.get(url, stream=True) as r:
                r.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)
                with open(local_zip_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
            print("Download complete.")

            # Extract the zip file
            with zipfile.ZipFile(local_zip_path, "r") as zip_ref:
                zip_ref.extractall(local_dir)
            print(f"Extracted files to {local_dir}")

            # Clean up the downloaded zip file
            os.remove(local_zip_path)

        except requests.exceptions.RequestException as e:
            print(f"Error downloading file: {e}")
            return None  # Or handle the error as needed
    else:
        print(f"Found local shapefile at {local_shp_path}. Loading...")

    # Load the shapefile using geopandas
    blocks = gp.read_file(local_shp_path)

    # Reproject the data to the desired coordinate reference system
    blocks = blocks.to_crs("EPSG:3078")

    return blocks


def deal_with_ann_arbor(df: gp.GeoDataFrame, candidate_cols: list, house2024):
    """
    Ann Arbor Township is a mess - it has "numerous enclaves" i.e. islands
    and, more importantly, is a precinct that is split into more than one
    legislative district. We will split it into two pieces and estimate
    vote totals by using relative area (not ideal; maybe we can use Census
    blocks later)
    """

    cutter = house2024.loc[house2024["SLDLST"] == "023", "geometry"].iloc[0]
    aa_1 = df.loc[df["Precinct_L"] == "Ann Arbor Township, Precinct 1"].clip(
        cutter
    )  # creates a new, 1 row dataframe
    cutter = house2024.loc[house2024["SLDLST"] == "048", "geometry"].iloc[0]
    aa_2 = df.loc[df["Precinct_L"] == "Ann Arbor Township, Precinct 1"].clip(cutter)

    aa_0 = df.loc[df["Precinct_L"] == "Ann Arbor Township, Precinct 1"]

    # Update the area of the new precincts
    aa_2.loc[:, "ShapeSTAre"] = aa_2["geometry"].area
    aa_1.loc[:, "ShapeSTAre"] = aa_1["geometry"].area

    # Update the border length of new precincts
    aa_2.loc[:, "ShapeSTLen"] = aa_2["geometry"].length
    aa_1.loc[:, "ShapeSTLen"] = aa_1["geometry"].length

    # Update the vote totals/registered voters based on area
    # The boundaries of this township do not appear to be based
    # on Census blocks, but we should confirm this
    # TODO: check whether we can use Census blocks instead
    aa_1_ratio = aa_1.loc[:, "ShapeSTAre"].iloc[0] / aa_0.loc[:, "ShapeSTAre"].iloc[0]
    aa_2_ratio = aa_2.loc[:, "ShapeSTAre"].iloc[0] / aa_0.loc[:, "ShapeSTAre"].iloc[0]

    aa_1.loc[:, candidate_cols] *= aa_1_ratio
    aa_2.loc[:, candidate_cols] *= aa_2_ratio

    # Let's pretend that the new precincts are "wards". And assign them to the
    # right state legislative district
    aa_1.loc[:, "WARD"] = "1"
    aa_1.loc[:, "WardNumber"] = 1
    aa_1.loc[:, "SLDLST"] = "023"
    aa_1.loc[:, "SLDUST"] = "015"

    aa_2.loc[:, "WARD"] = "2"
    aa_2.loc[:, "WardNumber"] = 2
    aa_2.loc[:, "SLDLST"] = "048"
    aa_2.loc[:, "SLDUST"] = "014"

    # Drop old Ann Arbor Township row
    df = df[df["Precinct_L"] != "Ann Arbor Township, Precinct 1"]

    # Add two new rows
    return pd.concat([df, aa_1, aa_2])


def load_data():
    votes, candidate_columns = load_and_format_votes()
    votes = reallocate_detroit_counting_board_votes(votes, candidate_columns)

    shapefiles = load_and_format_precincts_shapefile()
    df = merge_shapefiles_and_votes(votes, shapefiles)

    house2024, senate2024, house_subset, senate_subset = load_legislature(df)

    # Now look at Census shapefiles (for official population estimates)
    blocks = load_tiger_blocks()
    df["unique_precinct"] = (
        df["CountyCode"].astype(str)
        + "_"
        + df["CityTownName"].astype(str)
        + "_"
        + df["WardNumber"].astype(str)
        + "_"
        + df["PrecinctNumber"].astype(str)
    )
    data = dissolve_small_into_large(blocks, df, identifier_column="unique_precinct")

    # Assign each precinct to a SLDLST and SLDUST
    data = label_small_with_large(
        data, house2024[["geometry", "SLDLST"]], identifier_column="SLDLST"
    )

    data = label_small_with_large(
        data, senate2024[["geometry", "SLDUST"]], identifier_column="SLDUST"
    )

    data = deal_with_ann_arbor(
        data,
        candidate_cols=candidate_columns + ["Registered", "Active_Vot"],
        house2024=house2024,
    )

    data["unique_precinct"] = (
        data["CountyCode"].astype(str)
        + "_"
        + data["CityTownName"].astype(str)
        + "_"
        + data["WardNumber"].astype(str)
        + "_"
        + data["PrecinctNumber"].astype(str)
    )
    return data, house2024, senate2024, house_subset, senate_subset


if __name__ == "__main__":
    # Load voting results
    data = load_data()
    print(f"Data ({len(data)} precincts) loaded successfully!")
