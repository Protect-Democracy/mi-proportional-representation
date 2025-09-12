"""
This module contains the core logic for the redistricting simulation.
"""

import itertools
import pickle
import random
from functools import partial
from typing import Any, Dict, List, Tuple

import libpysal as lp
import networkx as nx
import numpy as np
import pandas as pd
from gerrychain import Graph, MarkovChain, Partition, accept
from gerrychain.constraints import contiguous
from gerrychain.proposals import recom
from gerrychain.tree import bipartition_tree
from gerrychain.updaters import Tally, cut_edges
from joblib import Memory
from scipy.stats import norm
from tqdm import trange

from constants import (
    RECOM_EPSILON,
    RECOM_REGION_SURCHARGE,
)
from data_loading import load_data
from utils import fit_statistical_models, modify_2party_share

# --- Constants and Configuration ---

# Seed for reproducibility
random.seed(101)
np.random.seed(101)

# Caching configuration
CACHE_LOCATION = "./.joblib_cache"
memory = Memory(CACHE_LOCATION, verbose=0)

# Simulation parameters
N_MAPS = 1000
HOUSE_SEATS_TOTAL = 100
SENATE_SEATS_TOTAL = 38
MMP_TOTAL_SEATS = 138
MMP_PROPORTIONAL_SEATS = 38
OL5_HOUSE_DISTRICTS = 20
OL5_SENATE_DISTRICTS = 7
OL5_DISTRICT_MAGNITUDE = 5


# --- Core Graph and Data Functions ---


def load_and_prepare_graph_() -> Tuple[nx.Graph, pd.DataFrame]:
    """
    Loads precinct data, constructs a NetworkX graph, and connects islands.
    This function is cached to avoid expensive repeated data loading and graph construction.

    Returns:
        A tuple containing the connected NetworkX graph and the original DataFrame.
    """
    print("Loading data and preparing graph...")
    data, house2024, senate2024, _, _ = load_data()

    # Create graph from GeoDataFrame adjacency
    gdf_neighbors = lp.weights.Queen.from_dataframe(data, use_index=False)
    neighbor_matrix, _ = gdf_neighbors.full()
    graph = nx.from_numpy_array(
        neighbor_matrix, nodelist=data["unique_precinct"].values
    )

    # Add node attributes
    attributes_dict = {
        row["unique_precinct"]: row.to_dict()
        for _, row in data[
            ["CityTownName", "unique_precinct", "POP20", "SLDLST"]
        ].iterrows()
    }
    nx.set_node_attributes(graph, attributes_dict)

    # Manually add edges to connect islands for GerryChain contiguity checks
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
    graph.add_edges_from(edges_to_add)

    if not nx.is_connected(graph):
        raise RuntimeError("Graph is not connected after adding manual edges.")
    print("Graph is connected.")
    return graph, data, house2024, senate2024


# --- Gerrymandering and Map Drawing ---


def draw_new_districts_(
    graph: nx.Graph, num_districts: int, num_steps: int
) -> List[Dict[Any, int]]:
    """
    Uses GerryChain to generate a sequence of district maps.

    Args:
        graph: The NetworkX graph of precincts.
        num_districts: The number of districts to create.
        num_steps: The number of steps in the Markov chain (number of maps).

    Returns:
        A list of assignment dictionaries, where each dictionary maps a node
        to a district ID.
    """
    graph_gerry = Graph.from_networkx(graph)
    pop_col = "POP20"
    updaters = {
        "population": Tally(pop_col, alias="population"),
        "cut_edges": cut_edges,
    }

    assignments = []
    for _ in trange(num_steps):
        try:
            initial_partition = Partition.from_random_assignment(
                graph_gerry,
                n_parts=num_districts,
                epsilon=RECOM_EPSILON,
                pop_col=pop_col,
                updaters=updaters,
            )

            ideal_population = sum(initial_partition["population"].values()) / len(
                initial_partition
            )

            proposal = partial(
                recom,
                pop_col=pop_col,
                pop_target=ideal_population,
                epsilon=RECOM_EPSILON,
                node_repeats=100,
                region_surcharge={"CityTownName": RECOM_REGION_SURCHARGE},
                method=partial(
                    bipartition_tree, max_attempts=100, allow_pair_reselection=True
                ),
            )

            chain = MarkovChain(
                proposal=proposal,
                constraints=[contiguous],
                accept=accept.always_accept,
                initial_state=initial_partition,
                total_steps=1,
            )
            for i, p in enumerate(chain):
                if i == 0:
                    assignments.append(p.assignment)
        except Exception:
            pass

    return assignments


def collapse_graph_by_label(graph: nx.Graph, label_key: str) -> nx.Graph:
    """Collapses a graph by a given node attribute."""
    new_graph = nx.Graph()
    for node, data in graph.nodes(data=True):
        label = data[label_key]
        if label not in new_graph:
            new_graph.add_node(label)
        for neighbor in graph.neighbors(node):
            neighbor_label = graph.nodes[neighbor][label_key]
            if label != neighbor_label:
                new_graph.add_edge(label, neighbor_label)
    return new_graph


def merge_two_nodes(graph: nx.Graph, node1: Any, node2: Any) -> nx.Graph:
    """Merges two nodes in a graph."""
    new_graph = graph.copy()
    neighbors = set(new_graph.neighbors(node1)) | set(new_graph.neighbors(node2))
    neighbors.discard(node1)
    neighbors.discard(node2)

    new_node_id = f"{node1}-{node2}"
    contains1 = new_graph.nodes[node1].get("contains", [node1])
    contains2 = new_graph.nodes[node2].get("contains", [node2])

    new_graph.add_node(new_node_id, contains=contains1 + contains2)
    for neighbor in neighbors:
        new_graph.add_edge(new_node_id, neighbor)

    new_graph.remove_nodes_from([node1, node2])
    return new_graph


def create_variable_magnitude_districts_(
    graph: nx.Graph, data: pd.DataFrame, base_assignment: Dict, min_size: int = 3
) -> Tuple[List[Any], List[int]]:
    """Combines single-member districts into larger, variable-sized districts."""
    graph_with_districts = graph.copy()
    nx.set_node_attributes(
        graph_with_districts,
        {n: str(d) for n, d in base_assignment.items()},
        "initial_district",
    )

    collapsed_graph = collapse_graph_by_label(graph_with_districts, "initial_district")
    nx.set_node_attributes(
        collapsed_graph, {n: [n] for n in collapsed_graph.nodes}, "contains"
    )

    while True:
        contents = nx.get_node_attributes(collapsed_graph, "contains")
        magnitudes = {k: len(v) for k, v in contents.items()}
        small_nodes = [n for n, size in magnitudes.items() if size < min_size]

        if not small_nodes:
            break

        node1 = random.choice(small_nodes)
        neighbors = list(collapsed_graph.neighbors(node1))
        if not neighbors:
            print(f"Warning: Node {node1} has no neighbors to merge with. Skipping.")
            break

        node2 = random.choice(neighbors)
        collapsed_graph = merge_two_nodes(collapsed_graph, node1, node2)

    final_contents = nx.get_node_attributes(collapsed_graph, "contains")
    district_magnitudes = [len(v) for v in final_contents.values()]

    initial_to_large_map = {
        initial_dist: large_dist
        for large_dist, initial_list in final_contents.items()
        for initial_dist in initial_list
    }

    final_assignment = [
        initial_to_large_map[str(base_assignment[p])] for p in data["unique_precinct"]
    ]

    return final_assignment, district_magnitudes


def _get_maps_for_scenario_(
    graph: nx.Graph, data: pd.DataFrame, scenario: str, n_maps: int
) -> Dict[str, Any]:
    """Helper function to generate district maps for a given scenario."""
    results: Dict[str, Any] = {"House": {}, "Senate": None}

    if scenario in ["MMP", "OL1"]:
        assignments = draw_new_districts(
            graph, num_districts=HOUSE_SEATS_TOTAL, num_steps=n_maps
        )
        results["House"] = {
            "assignment": assignments,
            "district_magnitudes": n_maps * [[1] * HOUSE_SEATS_TOTAL],
        }
        if scenario == "OL1":
            senate_assignments = draw_new_districts(
                graph, num_districts=SENATE_SEATS_TOTAL, num_steps=n_maps
            )
            results["Senate"] = {
                "assignment": senate_assignments,
                "district_magnitudes": n_maps * [[1] * SENATE_SEATS_TOTAL],
            }
    elif scenario == "OL5":
        house_assignments = draw_new_districts(
            graph, num_districts=OL5_HOUSE_DISTRICTS, num_steps=n_maps
        )
        senate_assignments = draw_new_districts(
            graph, num_districts=OL5_SENATE_DISTRICTS, num_steps=n_maps
        )
        results["House"] = {
            "assignment": house_assignments,
            "district_magnitudes": n_maps
            * [[OL5_DISTRICT_MAGNITUDE] * OL5_HOUSE_DISTRICTS],
        }
        results["Senate"] = {
            "assignment": senate_assignments,
            "district_magnitudes": n_maps
            * [[OL5_DISTRICT_MAGNITUDE] * OL5_SENATE_DISTRICTS],
        }
    elif scenario == "OL9":
        base_maps = get_district_maps(graph, data, "MMP", n_maps)
        single_member_house = base_maps["House"]["assignment"]
        single_member_senate = draw_new_districts(
            graph, num_districts=SENATE_SEATS_TOTAL, num_steps=n_maps
        )

        house_assignments, house_mags = [], []
        senate_assignments, senate_mags = [], []

        for i in trange(n_maps, desc="Aggregating OL9 districts"):
            try:
                assignment_h, dist_mag_h = create_variable_magnitude_districts(
                    graph, data, single_member_house[i]
                )
                assignment_s, dist_mag_s = create_variable_magnitude_districts(
                    graph, data, single_member_senate[i]
                )
                unique_precincts = list(single_member_house[i].keys())
                house_assignments.append(
                    {
                        unique_precincts[j]: assignment_h[j]
                        for j in range(len(unique_precincts))
                    }
                )
                house_mags.append(dist_mag_h)
                senate_assignments.append(
                    {
                        unique_precincts[j]: assignment_s[j]
                        for j in range(len(unique_precincts))
                    }
                )
                senate_mags.append(dist_mag_s)
            except Exception as e:
                print(f"Warning: Failed to create OL9 district for map {i}. Error: {e}")
                continue

        results["House"] = {
            "assignment": house_assignments,
            "district_magnitudes": house_mags,
        }
        results["Senate"] = {
            "assignment": senate_assignments,
            "district_magnitudes": senate_mags,
        }
    return results


def get_district_maps(
    graph: nx.Graph, data: pd.DataFrame, scenario: str, n_maps: int
) -> Dict[str, Any]:
    """
    A cached wrapper for generating district maps for a given scenario.

    Args:
        graph: The NetworkX precinct graph.
        data: The precinct DataFrame.
        scenario: The electoral scenario name (e.g., "MMP", "OL5").
        n_maps: The number of maps to generate.

    Returns:
        A dictionary containing lists of assignments and district magnitudes for House and Senate.
    """
    print(f"Generating {n_maps} maps for scenario: {scenario}...")
    return _get_maps_for_scenario(graph, data, scenario, n_maps)


# --- Vote Simulation and Seat Allocation ---


def jefferson_method(
    votes: Dict[str, float], allocated_seats: Dict[str, int], seats_to_allocate: int
) -> Dict[str, int]:
    """Allocates seats using the Jefferson/D'Hondt method."""
    seats = allocated_seats.copy()
    parties = [p for p, v in votes.items() if v > 0]
    if not parties:
        return seats

    for _ in range(seats_to_allocate):
        quotients = {p: votes[p] / (seats[p] + 1) for p in parties}
        most_deserving = max(quotients, key=quotients.get)
        seats[most_deserving] += 1
    return seats


def simulate_district_votes(
    district_df: pd.DataFrame,
    models: Dict,
    n_parties: int,
    partisan_trend: str,
    crossover_method="district",
) -> np.ndarray:
    """
    Simulates party vote shares within districts based on presidential vote,
    partisan trend, ticket-splitting, and number of parties.
    """
    trend_alpha = {"more_gop": -0.2, "more_dem": 0.2, "no_change": 0.0}[partisan_trend]
    gop_share = modify_2party_share(district_df["2partyshare"].values, trend_alpha)

    # ticket-splitting should be done at the district - not precinct level,
    # because candidate quality is a district-level feature
    if crossover_method == "district":
        crossover_model = models["crossover"]
        # 1. Identify the unique districts.
        unique_districts = district_df["district"].unique()
        n_districts = len(unique_districts)

        # 2. Generate one random modification value for each unique district.
        district_modifications = norm(
            loc=crossover_model.x[0], scale=crossover_model.x[1]
        ).rvs(size=n_districts)

        # 3. Create a map to associate each district with its random modification value.
        modification_map = pd.Series(district_modifications, index=unique_districts)

        # 4. Apply the district-level modification to each precinct.
        # All precincts in the same district will receive the same modification.
        precinct_level_modification = (
            district_df["district"].map(modification_map).values
        )

        gop_share = np.clip(gop_share + precinct_level_modification, 0, 1).reshape(
            -1, 1
        )
    else:
        gop_share = np.clip(gop_share, 0, 1).reshape(-1, 1)

    # GOP + Dem add up to 1 in any case
    dem_share = np.clip(1.0 - gop_share, 0, 1).reshape(-1, 1)

    if n_parties == 2:
        return np.concatenate([dem_share, gop_share], axis=1)

    def progressive_fraction(x):
        return 0.1 + (1.0 - x) * (0.65 - 0.1)

    def maga_fraction(x):
        return 0.1 + x * (0.72 - 0.1)

    prog_votes = np.array(progressive_fraction(gop_share) * dem_share).reshape(-1, 1)
    dem_votes = np.array((1.0 - progressive_fraction(gop_share)) * dem_share).reshape(
        -1, 1
    )
    gop_votes = np.array((1.0 - maga_fraction(gop_share)) * gop_share).reshape(-1, 1)
    maga_votes = np.array(maga_fraction(gop_share) * gop_share).reshape(-1, 1)

    return np.concatenate([prog_votes, dem_votes, gop_votes, maga_votes], axis=1)


def _allocate_seats(
    simulated_district_votes: np.ndarray,
    party_names: List[str],
    scenario: Dict,
    district_magnitudes: List[int],
) -> Dict[str, int]:
    """Helper function to allocate seats based on the scenario."""
    total_seats = {p: 0 for p in party_names}
    scenario_name = scenario["scenario_name"]

    if scenario_name in ["OL1", "MMP"]:
        for row in simulated_district_votes:
            winner = party_names[np.argmax(row)]
            total_seats[winner] += 1
        if scenario_name == "MMP":
            statewide_votes = np.sum(simulated_district_votes, axis=0)
            minimum_seats = np.floor(
                MMP_TOTAL_SEATS * statewide_votes / statewide_votes.sum()
            )
            for i, party in enumerate(party_names):
                if total_seats[party] < minimum_seats[i]:
                    total_seats[party] = int(minimum_seats[i])
            seats_to_allocate = int(MMP_TOTAL_SEATS - sum(total_seats.values()))
            if seats_to_allocate > 0:
                vote_dict = {p: statewide_votes[i] for i, p in enumerate(party_names)}
                total_seats = jefferson_method(
                    vote_dict, total_seats, seats_to_allocate
                )
    elif scenario_name in ["OL5", "OL9"]:
        for i, row in enumerate(simulated_district_votes):
            dist_mag = (
                OL5_DISTRICT_MAGNITUDE
                if scenario_name == "OL5"
                else district_magnitudes[i]
            )
            vote_dict = {p: row[j] for j, p in enumerate(party_names)}
            district_seats = jefferson_method(
                vote_dict, {p: 0 for p in party_names}, dist_mag
            )
            for party, seats in district_seats.items():
                total_seats[party] += seats
    return {p: int(v) for p, v in total_seats.items()}


def run_election(
    assignment: List,
    data: pd.DataFrame,
    models: Dict,
    scenario: Dict,
    district_magnitudes: List[int] = None,
    return_districts=False,
) -> Dict[str, int]:
    """
    Calculates the legislative seat breakdown for a given map and scenario.
    """
    districts = data.copy()
    districts["district"] = districts["unique_precinct"].map(dict(assignment))

    party_names = {2: ["D", "R"], 4: ["P", "D", "R", "M"]}[scenario["n_parties"]]

    simulated_precinct_votes = simulate_district_votes(
        districts, models, scenario["n_parties"], scenario["partisan_trend"]
    )
    total_legislator_votes = np.array(
        districts["STATE_REP_GOP"].values + districts["STATE_REP_DEM"].values
    ).reshape(-1, 1)
    districts[party_names] = simulated_precinct_votes * np.repeat(
        total_legislator_votes, repeats=scenario["n_parties"], axis=1
    )
    grouped = districts.groupby("district")[
        party_names + ["KAMALA D. HARRIS", "DONALD J. TRUMP"]
    ].sum()
    simulated_district_votes = grouped[party_names].values

    total_seats = _allocate_seats(
        simulated_district_votes, party_names, scenario, district_magnitudes
    )

    if return_districts:
        winners = [party_names[np.argmax(row)] for row in simulated_district_votes]
        return total_seats, grouped, winners

    return total_seats


# Manually set module names for consistent caching even with Jupyter
load_and_prepare_graph_.__module__ = "pro-rep.module"
load_and_prepare_graph = memory.cache(load_and_prepare_graph_)

draw_new_districts_.__module__ = "pro-rep.module"
draw_new_districts = memory.cache(draw_new_districts_, ignore=["graph"])

create_variable_magnitude_districts_.__module__ = "pro-rep.module"
create_variable_magnitude_districts = memory.cache(
    create_variable_magnitude_districts_, ignore=["graph"]
)

_get_maps_for_scenario_.__module__ = "pro-rep.module"
_get_maps_for_scenario = memory.cache(_get_maps_for_scenario_, ignore=["graph", "data"])


def main():
    """Main function to run the entire redistricting simulation."""
    graph, data, house2024, senate2024 = load_and_prepare_graph()
    data["2partyshare"] = data["DONALD J. TRUMP"] / (
        data["KAMALA D. HARRIS"] + data["DONALD J. TRUMP"]
    )
    models = fit_statistical_models(data)

    all_results = {}
    # There are a bunch of scenarios to investigate. Let's organize them:
    scenarios = {
        "electoral_system": ["MMP", "OL1", "OL5", "OL9", "MMP"],
        "partisan_trend": ["more_gop", "more_dem", "no_change"],
        "n_parties": [2, 4],
    }

    all_results = {}
    # Get all combinations
    for combination in itertools.product(*scenarios.values()):
        scenario = dict(zip(scenarios.keys(), combination))
        print(
            f"--- Running: Scenario={scenario['electoral_system']}, Trend={scenario['partisan_trend']}, Parties={scenario['n_parties']} ---"
        )

        electoral_system = scenario["electoral_system"]
        partisan_trend = scenario["partisan_trend"]
        n_parties = scenario["n_parties"]
        # Create nested structure if it doesn't exist
        if electoral_system not in all_results:
            all_results[electoral_system] = {}

        if partisan_trend not in all_results[electoral_system]:
            all_results[electoral_system][partisan_trend] = {}

        if n_parties not in all_results[electoral_system][partisan_trend]:
            all_results[electoral_system][partisan_trend][n_parties] = {}
        all_results[scenario["electoral_system"]][scenario["partisan_trend"]][
            scenario["n_parties"]
        ] = {}

        the_maps = get_district_maps(
            graph, data, scenario["electoral_system"], n_maps=N_MAPS
        )

        scenario_config = {
            "scenario_name": scenario["electoral_system"],
            "n_parties": scenario["n_parties"],
            "partisan_trend": scenario["partisan_trend"],
        }

        house_results = []
        for i in trange(
            len(the_maps["House"]["assignment"]), desc="Simulating House maps"
        ):
            house_assignment = the_maps["House"]["assignment"][i]
            house_mags = the_maps["House"]["district_magnitudes"]

            house_seats = run_election(
                house_assignment,
                data,
                models,
                scenario_config,
                district_magnitudes=house_mags[i] if house_mags else None,
            )
            house_results.append(house_seats)

        if scenario["electoral_system"] in ["OL5", "OL9", "OL1"]:
            for i in trange(
                len(the_maps["Senate"]["assignment"]),
                desc="Simulating Senate maps",
            ):
                senate_assignment = the_maps["Senate"]["assignment"][i]
                senate_mags = the_maps["Senate"]["district_magnitudes"]

                senate_seats = run_election(
                    senate_assignment,
                    data,
                    models,
                    scenario_config,
                    district_magnitudes=senate_mags[i] if senate_mags else None,
                )

                if i < len(house_results):
                    for party, seats in senate_seats.items():
                        house_results[i][party] = house_results[i].get(party, 0) + seats

        all_results[scenario["electoral_system"]][scenario["partisan_trend"]][
            scenario["n_parties"]
        ] = house_results

    with open("simulation_results.pkl", "wb") as f:
        pickle.dump(all_results, f)
    print("\nSimulation complete. Results saved to simulation_results.pkl")


if __name__ == "__main__":
    main()
