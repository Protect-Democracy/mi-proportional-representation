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

from src.constants import (
    NEW_PARTY_SHOR_MCCARTY_THRESHOLD,
    RECOM_EPSILON,
    RECOM_REGION_SURCHARGE,
    TWO_PARTY_SHOR_MCCARTY_THRESHOLD,
)
from src.data_loading import (
    load_and_format_precincts_shapefile,
    load_and_format_votes,
    load_legislature,
    load_tiger_blocks,
    merge_shapefiles_and_votes,
    reallocate_detroit_counting_board_votes,
)
from src.redistricting import draw_new_districts
from src.utils import dissolve_small_into_large, label_small_with_large


def scenario1():
    return


def scenario2():
    return


def scenario3():
    return


if __name__ == "__main__":
    # Load voting results

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
        # If any party has won more seats in districts than the minimum number
        # of seats they should earn, then increase the total number of list
        # seats until that party’s district seats equals its minimum number
        # of seats, then rerun the minimum number of seats calculation for
        # each party, but replacing the number 139 with the sum of the new
        # number of list seats plus the number of districts seats plus one.
        if (
            assigned_seats["R"] > minimum_seats[1]
            or assigned_seats["D"] > minimum_seats[0]
        ):
            jefferson_method()  # not yet implemented, but this is extremely rare

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
