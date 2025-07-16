import random
from functools import partial

import libpysal as lp
import networkx as nx
import numpy as np
import pandas as pd
from gerrychain import Graph, MarkovChain, Partition, accept
from gerrychain.constraints import contiguous
from gerrychain.proposals import recom
from gerrychain.tree import bipartition_tree
from gerrychain.updaters import Tally, cut_edges
from scipy.optimize import minimize
from scipy.stats import norm
from tqdm import tqdm

from src.constants import (
    RECOM_EPSILON,
    RECOM_REGION_SURCHARGE,
)
from src.data_loading import (
    load_data,
)

random.seed(101)


def draw_new_districts(G, n_parts=None, existing_seat_column_name=None, n_steps=1000):
    """Use GerryChain to step through a Monte Carlo sequence of different maps."""

    graph = Graph.from_networkx(G)
    if n_parts is not None:
        initial_partition = Partition.from_random_assignment(
            graph,
            epsilon=RECOM_EPSILON,
            pop_col="POP20",
            n_parts=n_parts,
            updaters={
                "population": Tally("POP20", alias="population"),
                "cut_edges": cut_edges,
            },
        )

    elif existing_seat_column_name is not None:
        initial_partition = Partition(
            graph,
            existing_seat_column_name,
            n_parts=n_parts,
            updaters={
                "population": Tally("POP20", alias="population"),
                "cut_edges": cut_edges,
            },
        )
    else:
        raise Exception(
            "You must specify either an existing set of districts, or the number of new districts"
        )
    ideal_population = sum(initial_partition["population"].values()) / len(
        initial_partition
    )

    proposal = partial(
        recom,
        pop_col="POP20",
        pop_target=ideal_population,
        epsilon=RECOM_EPSILON,
        node_repeats=100,
        region_surcharge={"CityTownName": RECOM_REGION_SURCHARGE},
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
    graph of precincts for contiguous districts. This function adds some edges
    by hand that connect islands to the mainland through their most
    appropriate nearest precincts.
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


def graph_from_df(df):
    """
    Turn a GeoDataFrame into a NetworkX graph via its adjacency matrix
    """

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


def jefferson_method(votes_received, allocated_seats, seats_to_allocate):
    """The Jefferson/D'Hondt/Greatest Divisors method of allocating seats."""

    while seats_to_allocate > 0:
        quotient = {}
        for party in allocated_seats.keys():
            quotient[party] = votes_received[party] / (allocated_seats[party] + 1)
        most_deserving_party = sorted(
            quotient.items(), key=lambda x: x[1], reverse=True
        )[0][0]
        allocated_seats[most_deserving_party] += 1
        seats_to_allocate -= 1

    return allocated_seats


def modify_2party_share(value, alpha):
    """
    Tweaking the 2-party share to allow for the state trending
    more red or blue. Alpha should be ~0.1 for a small trend
    """
    if alpha > 0:
        return value - alpha * (value - value**2)
    if alpha < 0:
        return value + alpha * (value - value**0.5)


def party_popularity(partisanship, result4):
    x = -1 * (partisanship * 4 - 2)
    # "progressive party"
    prog = result4.x[0] * norm(loc=result4.x[1], scale=result4.x[2]).pdf(x)
    # "democratic party"
    dem4 = result4.x[3] * norm(loc=result4.x[4], scale=result4.x[5]).pdf(x)
    # "republican party"
    gop4 = result4.x[6] * norm(loc=result4.x[7], scale=result4.x[8]).pdf(x)
    # "maga party"
    maga = result4.x[9] * norm(loc=result4.x[10], scale=result4.x[11]).pdf(x)

    raw4 = np.vstack([prog, dem4, gop4, maga]).T
    return raw4 / np.repeat(np.sum(raw4, axis=1).reshape(-1, 1), repeats=4, axis=1)


def allocate_seats(votes, initial_seats, district_magnitude, res, result4):
    """Simulate votes by applying crossover voting. Then put the number of
    seats earned into a district dictionary."""
    n_parties = len(initial_seats.keys())
    seats = initial_seats.copy()

    # Add or subtract votes from the state rep compared to POTUS
    modification = norm(loc=res.x[0], scale=res.x[1]).rvs()
    votes[:, 0] = votes[:, 0] + modification
    votes[:, 1] = votes[:, 1] - modification

    # if it's a 2-party system, no more modifications are needed
    if n_parties == 4:
        # In a 4-party system, we need to further break down the votes
        #
        popularity4 = party_popularity(votes[:, 0], result4=result4)

        # progressive/dem split
        prog = votes[:, 0] * popularity4[:, 0] / (popularity4[:, 0] + popularity4[:, 1])
        dem = votes[:, 0] * popularity4[:, 1] / (popularity4[:, 0] + popularity4[:, 1])

        # republican/maga split
        gop = votes[:, 1] * popularity4[:, 2] / (popularity4[:, 2] + popularity4[:, 3])
        maga = votes[:, 1] * popularity4[:, 3] / (popularity4[:, 2] + popularity4[:, 3])

        votes = np.hstack(
            [
                prog.reshape(-1, 1),
                dem.reshape(-1, 1),
                gop.reshape(-1, 1),
                maga.reshape(-1, 1),
            ]
        )

    # Loop through each district and assign seats proportionally based
    # on the number of votes received
    for i in range(votes.shape[0]):
        district_seats = jefferson_method(
            {k: votes[i, j] for j, k in enumerate(initial_seats.keys())},
            allocated_seats={k: 0 for k in initial_seats.keys()},
            seats_to_allocate=district_magnitude,
        )
        for party in initial_seats.keys():
            seats[party] += district_seats[party]

    return seats, votes


def mmp_scenario(districts, res, result4):
    """
    This scenario has 100 district-based seats, and 38 statewide proportional
    seats. The overall representation is mechanically forced to be proportional
    with some logic around how the 38 are allocated.
    """
    # First, allocate district-based seats
    district_magnitude = 1
    simulated_votes = np.vstack(
        [districts["2partyshare"], 1.0 - districts["2partyshare"]]
    ).reshape(-1, 2)

    if scenario_dictionary["n_parties"] == 2:
        seats = {
            "D": 0,
            "R": 0,
        }

    else:
        seats = {"P": 0, "D": 0, "R": 0, "M": 0}
    seats, votes = allocate_seats(
        simulated_votes,
        seats,
        district_magnitude=district_magnitude,
        res=res,
        result4=result4,
    )

    if sum(seats.values()) > 138:
        print("over 138")
    # Now, compute the statewide proportional seats

    # in the MMP scenario, there are 38 additional seats allocated on
    # a statewide basis. From the requirements:

    # For any viable party that earned fewer seats in districts than its
    # minimum number of seats, it wins a number of list seats
    # equal to the difference.
    minimum_seats = np.floor(139 * np.sum(votes, axis=0) / np.sum(votes))

    for i, party in enumerate(seats.keys()):
        if seats[party] < minimum_seats[i]:
            difference = minimum_seats[i] - seats[party]
            seats[party] += difference
    # If, after that step, fewer than 38 list seats have been awarded,
    # determine the remaining seats using the Jefferson Method,
    # also called the greatest divisors method.
    seats_to_allocate = 138 - sum(seats.values())
    seats = jefferson_method(
        {k: np.sum(votes, axis=0)[i] for i, k in enumerate(seats.keys())},
        seats,
        seats_to_allocate=seats_to_allocate,
    )

    return seats


def ol5_scenario(districts, res, result4):
    """
    This is the simplest scenario: 20 districts of 5 seats each. No
    extra logic - just add up the votes for each party in each district
    """
    district_magnitude = 5
    simulated_votes = np.vstack(
        [districts["2partyshare"], 1.0 - districts["2partyshare"]]
    ).reshape(-1, 2)

    if scenario_dictionary["n_parties"] == 2:
        seats = {
            "D": 0,
            "R": 0,
        }

    else:
        seats = {"P": 0, "D": 0, "R": 0, "M": 0}

    seats, votes = allocate_seats(
        simulated_votes,
        seats,
        district_magnitude=district_magnitude,
        res=res,
        result4=result4,
    )
    return seats


def ol9_scenario():
    return


def legislature_given_map(
    assignment,
    df,
    scenario_dictionary,
    res,
    result4,
    verbose=False,
):
    """
    What is the breakdown of the seats in the legislature,
    given a particular map and scenario?
    """
    # Figure out the partisanship of each district
    districts = df.copy()
    districts["district"] = [
        assignment[unique_precinct] for unique_precinct in districts["unique_precinct"]
    ]
    grouped_districts = districts.groupby("district")[
        ["KAMALA D. HARRIS", "DONALD J. TRUMP", "STATE_REP_GOP", "STATE_REP_DEM"]
    ].sum()
    grouped_districts["2partyshare"] = grouped_districts["KAMALA D. HARRIS"] / (
        grouped_districts["KAMALA D. HARRIS"] + grouped_districts["DONALD J. TRUMP"]
    )
    # What does the scenario say about the partisanship trend of Michigan?
    if scenario_dictionary["partisan_trend"] == "more_gop":
        grouped_districts["2partyshare"] = modify_2party_share(
            grouped_districts["2partyshare"], 0.2
        )

    elif scenario_dictionary["partisan_trend"] == "more_dem":
        grouped_districts["2partyshare"] = modify_2party_share(
            grouped_districts["2partyshare"], -0.2
        )

    if scenario_dictionary["scenario_name"] == "MMP":
        seats = mmp_scenario(grouped_districts, res, result4)

    if scenario_dictionary["scenario_name"] == "OL5":
        seats = ol5_scenario(grouped_districts, res, result4)

    return seats


def two_component_gaussian(params, x):
    a1, mu1, sigma1, a2, mu2, sigma2 = params

    z = a1 * norm(loc=mu1, scale=sigma1).pdf(x) + a2 * norm(loc=mu2, scale=sigma2).pdf(
        x
    )
    return z


def four_component_gaussian(params, x):
    a1, mu1, sigma1, a2, mu2, sigma2, a3, mu3, sigma3, a4, mu4, sigma4 = params

    z = (
        a1 * norm(loc=mu1, scale=sigma1).pdf(x)
        + a2 * norm(loc=mu2, scale=sigma2).pdf(x)
        + a3 * norm(loc=mu3, scale=sigma3).pdf(x)
        + a4 * norm(loc=mu4, scale=sigma4).pdf(x)
    )
    return z


def fit_crossover(data):
    h = np.histogram(
        data["KAMALA D. HARRIS"] / (data["KAMALA D. HARRIS"] + data["DONALD J. TRUMP"])
        - data["STATE_REP_DEM"] / (data["STATE_REP_DEM"] + data["STATE_REP_GOP"]),
        bins=np.linspace(-0.5, 0.5, 201),
        density=True,
    )

    def norm_loss(params, x=h[1][:-1], y=h[0]):
        return np.mean(((norm.pdf(x, loc=params[0], scale=params[1])) - y) ** 2)

    res = minimize(norm_loss, x0=[0, 0.02], tol=1e-10)
    return res


def fit_shor_mccarty():
    # if there are >2 major parties, we can't get real estimates from
    # the election results. We have to make a toy model
    def two_component_loss(params, x, y):
        return np.sqrt(np.mean((two_component_gaussian(params, x) - y) ** 2))

    def four_component_loss(params, x, y):
        return np.sqrt(np.mean((four_component_gaussian(params, x) - y) ** 2))

    leg = pd.read_stata("raw_data/legislator_data/shor_mccarty.dta")
    h = np.histogram(leg["np_score"], bins=np.linspace(-2, 2, 101), density=True)
    result2 = minimize(
        two_component_loss,
        x0=np.array([0.5, -1.0, 0.5, 0.5, 0.8, 0.5]),  # initial guess
        args=(h[1][:-1], h[0]),
        tol=1e-8,
    )
    # Here, we explicitly determine some bounds on the parameters in order
    # to soft-prescribe which parties emerge in a 4-party system. Essentially,
    # we are assuming a progressive (far-left) and MAGA (far-right) party
    # will arise, and the traditional Democratic and Republican parties will
    # be closer to the center. (you can tweak the bounds here if you disagree)
    result4 = minimize(
        four_component_loss,
        x0=np.array(
            [0.25, -1.5, 0.4, 0.25, -0.8, 0.5, 0.25, 0.5, 0.4, 0.25, 1.0, 0.25]
        ),  # initial guess
        args=(h[1][:-1], h[0]),
        tol=1e-8,
        bounds=(
            [0.2, 0.5],
            [-2, -1.0],
            [0.1, 1.0],
            [0.2, 0.5],
            [-1.0, -0.5],
            [0.1, 1.0],
            [0.2, 0.5],
            [0.5, 1],
            [0.1, 1.0],
            [0.2, 0.5],
            [1, 2],
            [0.1, 1.0],
        ),
    )
    return result2, result4


if __name__ == "__main__":
    # Load voting results
    data, house2024, senate2024, house_subset, senate_subset = load_data()

    # Turn our dataframe into a NetworkX Graph
    G = graph_from_df(data)
    G = connect_graph(G)

    # Get estimates of crossover voting, from real 2024 data
    # as well as our 4-party model
    res = fit_crossover(data)
    result2, result4 = fit_shor_mccarty()

    # Loop through different electoral scenarios
    legislatures = {}
    district_seats = 100
    n_maps = 10
    # Big for loop to go through all scenarios
    for scenario in ["MMP", "OL5"]:
        legislatures[scenario] = {}

        if scenario == "MMP":
            district_magnitude = 1
        elif scenario == "OL5":
            district_magnitude = 5
        try:
            assignment_list_house = draw_new_districts(
                G,
                n_parts=int(district_seats / district_magnitude),
                n_steps=n_maps,
            )
            if scenario == "OL5":
                district_seats = 35
                assignment_list_senate = draw_new_districts(
                    G,
                    n_parts=int(district_seats / district_magnitude),
                    n_steps=n_maps,
                )

        except RuntimeError:
            # sometimes the map-drawing algorithm gets stuck.
            # It's okay to ignore this and just keep going
            pass

        for partisan_trend in ["no_change", "more_gop", "more_dem"]:
            legislatures[scenario][partisan_trend] = {}
            for n_parties in [2, 4]:
                legislatures[scenario][partisan_trend][n_parties] = []
                for assignment in tqdm(assignment_list_house):
                    scenario_dictionary = {
                        "scenario_name": scenario,
                        "n_parties": n_parties,
                        "partisan_trend": partisan_trend,
                    }
                    assigned_seats = legislature_given_map(
                        assignment,
                        data,
                        scenario_dictionary=scenario_dictionary,
                        res=res,
                        result4=result4,
                    )
                    legislatures[scenario][partisan_trend][n_parties].append(
                        assigned_seats
                    )
                if scenario == "OL5":
                    for i, assignment in tqdm(enumerate(assignment_list_senate)):
                        scenario_dictionary = {
                            "scenario_name": scenario,
                            "n_parties": n_parties,
                            "partisan_trend": partisan_trend,
                        }
                        assigned_seats_senate = legislature_given_map(
                            assignment,
                            data,
                            scenario_dictionary=scenario_dictionary,
                            res=res,
                            result4=result4,
                        )

                        for party in assigned_seats.keys():
                            legislatures[scenario][partisan_trend][n_parties][i][
                                party
                            ] += assigned_seats_senate[party]
