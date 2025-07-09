import random
from functools import partial

import geopandas as gp
import libpysal as lp
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from gerrychain import Graph, MarkovChain, Partition, accept
from gerrychain.constraints import contiguous
from gerrychain.proposals import recom
from gerrychain.tree import bipartition_tree
from gerrychain.updaters import Tally, cut_edges
from scipy.stats import gaussian_kde
from tqdm import tqdm

from src.data_loading import (
    load_and_format_precincts_shapefile,
    load_and_format_votes,
    load_legislature,
    load_tiger_blocks,
    merge_shapefiles_and_votes,
    reallocate_detroit_counting_board_votes,
)
from src.utils import dissolve_small_into_large, label_small_with_large

random.seed(101)


def draw_new_districts(G, existing_seat_column_name, n_steps=1000):
    """Under construction!!"""

    graph = Graph.from_networkx(G)
    initial_partition = Partition(
        graph,
        assignment=existing_seat_column_name,
        updaters={
            "population": Tally("POP20", alias="population"),
            "cut_edges": cut_edges,
        },
    )

    ideal_population = sum(initial_partition["population"].values()) / len(
        initial_partition
    )

    proposal = partial(
        recom,
        pop_col="POP20",
        pop_target=ideal_population,
        epsilon=0.05,
        node_repeats=100,
        region_surcharge={"CityTownName": 1.0},
        method=partial(
            bipartition_tree,
            max_attempts=100,
            allow_pair_reselection=True,  # <-- This is the only change
        ),
    )

    recom_chain = MarkovChain(
        proposal=proposal,
        constraints=[contiguous],
        accept=accept.always_accept,
        initial_state=initial_partition,
        total_steps=n_steps,
    )

    assignment_list = []

    for i, item in enumerate(recom_chain):
        print(f"Finished step {i + 1}/{len(recom_chain)}", end="\r")
        assignment_list.append(item.assignment)

    return assignment_list


def connect_graph(G):
    """
    Some of the precincts in Michigan are islands; we need a connected
    graph of precincts though. This function adds some edes by hand that
    connect islands to the mainland through their most appropriate nearest
    precincts.
    """
    edges_to_add = [
        ("82_GROSSE ILE TOWNSHIP_0.0_1", "82_WYANDOTTE CITY_0.0_3"),
        ("82_GROSSE ILE TOWNSHIP_0.0_1", "82_RIVERVIEW CITY_0.0_1"),
        ("82_GROSSE ILE TOWNSHIP_0.0_1", "82_TRENTON CITY_0.0_1"),
        ("82_GROSSE ILE TOWNSHIP_0.0_2", "82_TRENTON CITY_0.0_4"),
        ("82_GROSSE ILE TOWNSHIP_0.0_3", "82_TRENTON CITY_0.0_4"),
        ("82_GROSSE ILE TOWNSHIP_0.0_3", "82_GIBRALTAR CITY_0.0_1"),
        ("15_PEAINE TOWNSHIP_0.0_12", "15_CHARLEVOIX TOWNSHIP_0.0_4"),
        ("15_PEAINE TOWNSHIP_0.0_12", "15_CHARLEVOIX CITY_1.0_18"),
        ("15_PEAINE TOWNSHIP_0.0_12", "15_CHARLEVOIX CITY_2.0_19"),
        ("49_BOIS BLANC TOWNSHIP_0.0_1", "49_MACKINAC ISLAND CITY_0.0_1"),
        ("49_MACKINAC ISLAND CITY_0.0_1", "49_ST IGNACE CITY_0.0_1"),
        ("24_WAWATAM TOWNSHIP_0.0_1", "49_MORAN TOWNSHIP_0.0_1"),
    ]
    for edge in edges_to_add:
        G.add_edge(edge[0], edge[1])

    if nx.is_connected(G):
        print("Graph is connected")
    else:
        print("Graph is disconnected... something has gone wrong")
    return G


# Turning the GeoDataFrame into a NetworkX graph via its adjacency matrix
def graph_from_df(df):
    gdf_neighbors = lp.weights.Queen.from_dataframe(df)
    neighbor_matrix, neighbor_idx = gdf_neighbors.full()

    G = nx.from_numpy_array(neighbor_matrix, nodelist=data["unique_precinct"].values)
    attributes_dict = {
        row["unique_precinct"]: row.to_dict()
        for index, row in data[
            ["CityTownName", "unique_precinct", "POP20", "SLDLST"]
        ].iterrows()
    }

    # Set the node attributes
    nx.set_node_attributes(G, attributes_dict)
    return G


def deal_with_ann_arbor(df: gp.GeoDataFrame, candidate_cols: list):
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
    aa_1.loc[:, "WardNumber"] = "1"
    aa_1.loc[:, "SLDLST"] = "023"
    aa_1.loc[:, "SLDUST"] = "015"

    aa_2.loc[:, "WARD"] = "2"
    aa_2.loc[:, "WardNumber"] = "2"
    aa_2.loc[:, "SLDLST"] = "048"
    aa_2.loc[:, "SLDUST"] = "014"

    # Drop old Ann Arbor Township row
    df = df[df["Precinct_L"] != "Ann Arbor Township, Precinct 1"]

    # Add two new rows
    return pd.concat([df, aa_1, aa_2])


def build_kde_sampler(x_values, y_values, bandwidth=None):
    """
    Build a callable that, given new x, samples y based on empirical (x, y) data.

    Args:
        x_values: array-like of shape (n_samples,)
        y_values: array-like of shape (n_samples,)
        bandwidth: optional bandwidth parameter for gaussian_kde

    Returns:
        sample_y(x_query): function that samples y given x_query
    """
    data = np.vstack([x_values, y_values])
    kde = gaussian_kde(data, bw_method=bandwidth)

    def sample_y(x_query, n_candidates=1000):
        """
        Sample y given x_query by rejection sampling.

        Args:
            x_query: float or array-like of shape (n_queries,)
            n_candidates: number of candidate y samples to draw

        Returns:
            array of sampled y's corresponding to x_query
        """
        x_query = np.atleast_1d(x_query)
        sampled_ys = []

        for xq in x_query:
            # Sample candidate ys from observed y's + jitter
            y_candidates = np.random.choice(
                y_values, size=n_candidates, replace=True
            ) + np.random.normal(scale=np.std(y_values) * 0.1, size=n_candidates)

            # Evaluate joint KDE at (xq, y_candidates)
            xy = np.vstack([np.full(n_candidates, xq), y_candidates])
            densities = kde.evaluate(xy)

            # Normalize densities to sum to 1 to use as probabilities
            probs = densities / np.sum(densities)

            # Sample a y from the candidate pool according to probs
            y_sampled = np.random.choice(y_candidates, p=probs)
            sampled_ys.append(y_sampled)

        return np.array(sampled_ys)

    return sample_y


def jefferson_method(votes, initial_allocations, remaining_seats_to_allocate):
    """Under construction"""
    print("hello world")


def legislature_given_map(assignment, df, sampler):
    """
    What is the breakdown of the seats in the legislature,
    given a particular map?
    """
    districts = df.copy()
    districts["district"] = [
        assignment[unique_precinct] for unique_precinct in districts["unique_precinct"]
    ]
    grouped_districts = districts.groupby("district")[
        ["KAMALA D. HARRIS", "DONALD J. TRUMP"]
    ].sum()
    grouped_districts["2partyshare"] = grouped_districts["DONALD J. TRUMP"] / (
        grouped_districts["KAMALA D. HARRIS"] + grouped_districts["DONALD J. TRUMP"]
    )

    # Figure out partisanship of each district with a groupby
    # Look up each partisanship in our lookup table
    seats = {
        "R": 0,
        "D": 0,
    }
    for value in grouped_districts["2partyshare"].values:
        if sampler(value) > 0:
            seats["R"] += 1
        else:
            seats["D"] += 1

    return seats


if __name__ == "__main__":
    # Load voting results
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
        data, candidate_cols=candidate_columns + ["Registered", "Active_Vot"]
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
    # Turn our dataframe into a NetworkX Graph
    G = graph_from_df(data)
    G = connect_graph(G)

    sampler = build_kde_sampler(
        np.hstack([senate_subset["2partyshare"], house_subset["2partyshare"]]),
        np.hstack([senate_subset["np_score"], house_subset["np_score"]]),
        bandwidth=0.4,
    )

    # District seats
    assignment_list = draw_new_districts(
        G, existing_seat_column_name="SLDLST", n_steps=1000
    )
    district_legislatures = []

    total_votes = data["KAMALA D. HARRIS"].sum() + data["DONALD J. TRUMP"].sum()
    party_votes = np.array(
        [data["KAMALA D. HARRIS"].sum(), data["DONALD J. TRUMP"].sum()]
    )
    partyshare = data["DONALD J. TRUMP"].sum() / (
        data["KAMALA D. HARRIS"].sum() + data["DONALD J. TRUMP"].sum()
    )

    # "Determine the minimum number of seats each party should earn by multiplying
    # the total number of votes cast for that party by 139 and dividing by the
    # total number of votes cast for all viable parties, ignoring any remainder."
    minimum_seats = np.floor(139 * party_votes / total_votes)

    for assignment in tqdm(assignment_list):
        assigned_seats = legislature_given_map(assignment, data, sampler)
        # For any viable party that earned fewer seats in districts than its
        # minimum number of seats, it wins a number of list seats
        # equal to the difference.
        if assigned_seats["R"] < minimum_seats[1]:
            r_difference = minimum_seats[1] - assigned_seats["R"]
            assigned_seats["R"] += r_difference

        if assigned_seats["D"] < minimum_seats[0]:
            d_difference = minimum_seats[0] - assigned_seats["D"]
            assigned_seats["D"] += d_difference

        # If, after that step, fewer than 38 list seats have been awarded,
        # determine the remaining seats using the Jefferson Method,
        # also called the greatest divisors method.
        if d_difference + r_difference < 38:
            # update with jefferson method
            pass

        district_legislatures.append(assigned_seats)

    # overall legislature is the sum of parties

    leg_df = pd.DataFrame(district_legislatures)
    plt.hist(leg_df["D"], bins=range(50, 80), label="Democrat", alpha=0.7)
    plt.hist(leg_df["R"], bins=range(50, 80), label="Republican", alpha=0.7)
    plt.ylabel("Count")
    plt.xlabel("Seats in legislature")
    plt.legend()
    plt.show()
