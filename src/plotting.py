import matplotlib.pyplot as plt
import geopandas as gp
from utils import dissolve_small_into_large
from data_loading import (
    load_and_format_precincts_shapefile,
    load_legislature,
    load_and_format_votes,
    reallocate_detroit_counting_board_votes,
    merge_shapefiles_and_votes,
)


def district_heatmap(precincts, districts):
    fig = plt.figure(figsize=[20, 20])
    plt.rcParams["savefig.facecolor"] = "1a1b26"
    plt.rcParams["figure.facecolor"] = "1a1b26"
    plt.rcParams["axes.facecolor"] = "1a1b26"

    df = precincts.copy()
    df["2partyshare"] = df["DONALD J. TRUMP"] / (
        df["DONALD J. TRUMP"] + df["KAMALA D. HARRIS"]
    )
    df.plot(
        "2partyshare",
        ax=plt.gca(),
        cmap="coolwarm",
        vmin=0,
        vmax=1,
        linewidth=0.0,
        edgecolor="None",
    )
    districts.plot(edgecolor="k", linewidth=2.0, ax=plt.gca(), facecolor="None")
    plt.gca().set_axis_off()
    plt.tight_layout()
    # plt.savefig('house_maps.pdf')
    plt.show()


def partisanship_scatterplot(house_subset, senate_subset):
    plt.scatter(house_subset["np_score"], house_subset["2partyshare"], label="house")
    plt.scatter(senate_subset["np_score"], senate_subset["2partyshare"], label="senate")
    plt.xlabel("Shor-McCarty score")
    plt.ylabel("2024 2-party share")
    plt.legend()
    plt.show()


def pop_density_map(blocks):
    df = blocks.copy()
    df["popdensity"] = df["POP20"] / df["ALAND20"]
    fig = plt.figure(figsize=[20, 20])
    df.plot(
        "popdensity", cmap="YlGnBu", edgecolor="None", linewidth=0, vmin=0, vmax=1e-3
    )
    plt.gca().set_axis_off()
    plt.tight_layout()
    # plt.savefig("mi_pop_density.pdf")
    plt.show()


def party_share_plot(data):
    df = data.copy()
    df["2partyshare"] = df["DONALD J. TRUMP"] / (
        df["DONALD J. TRUMP"] + df["KAMALA D. HARRIS"]
    )
    fig = plt.figure(figsize=[15, 15])
    df.plot("2partyshare", ax=plt.gca(), cmap="coolwarm", vmin=0, vmax=1)
    plt.gca().axis(None)
    plt.show()


if __name__ == "__main__":
    # Load voting results
    votes, candidate_columns = load_and_format_votes()
    votes = reallocate_detroit_counting_board_votes(votes, candidate_columns)

    shapefiles = load_and_format_precincts_shapefile()
    df = merge_shapefiles_and_votes(votes, shapefiles)

    house2024, senate2024, house_subset, senate_subset = load_legislature()

    # Now look at Census shapefiles (for official population estimates)
    blocks = gp.read_file("raw_data/tiger_blocks/tl_2024_26_tabblock20.shp")
    blocks = blocks.to_crs("EPSG:3078")
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
    print(f"Data ({len(data)} precincts) loaded successfully!")

    party_share_plot(df)
    district_heatmap(df, house2024)
    district_heatmap(df, senate2024)
    partisanship_scatterplot(house_subset, senate_subset)
