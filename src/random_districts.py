import pandas as pd

import numpy as np
from tqdm import trange
import networkx as nx

from src.data_loading import (
    load_data,
)

from src.redistricting import graph_from_df, connect_graph


class Districter:
    def __init__(
        self,
        G,
        df_pop,
        num_districts,
        tolerance,
        min_reps,
        n_representatives,
        column="POP20",
    ):
        self.graph = G
        self.impossible = False
        self.tolerance = tolerance
        self.n = len(list(self.graph.nodes))
        self.n_iterations = 100
        self.n_representatives = n_representatives
        self.n_attempts = 20

        if num_districts > self.n - 1:
            raise Exception("Number of districts must be less than number of counties")
        else:
            self.num_districts = num_districts

        self.min_reps = min_reps
        # Maximum reps per district is however many the largest county on its own would get
        self.max_reps = round(
            self.n_representatives * np.max(self.metric) / np.sum(self.metric)
        )
        self.ideal_district_size = np.sum(self.metric) / self.n_representatives

        self.representativeness = (
            self.metric / np.sum(self.metric) * self.n_representatives
        )
        self.representativeness -= np.floor(self.representativeness)
        self.initialize_districts()

    def initialize_districts(self):
        self.impossible = False
        self.districts = np.zeros((self.n))

        # Randomly assign initial districts

        self.districts[
            np.random.choice(np.arange(self.n), size=self.num_districts, replace=False)
        ] = 1 + np.arange(self.num_districts)
        # Grow the components randomly until there are no district 0s left anymore
        while np.any(self.districts == 0):
            choice1 = np.random.choice(np.where(self.districts == 0)[0], size=1)
            neighbors = np.where(self.adj_matrix[choice1, :].flatten())[0]
            different_neighbors = neighbors[np.where(self.districts[neighbors] != 0)[0]]
            if len(different_neighbors) > 0:
                self.districts[choice1] = self.districts[
                    np.random.choice(different_neighbors, size=1)
                ]

    def collapse_graph_by_label(self, G):
        # Create a new graph
        new_G = nx.Graph()
        # Create a dictionary to map labels to nodes

        # Iterate over the nodes in the original graph
        for node in G.nodes():
            # Get the label for the current node
            label = G.nodes[node]["district"]

            # If this label hasn't been seen before, add a new node to the new graph
            if label not in new_G.nodes:
                new_G.add_node(label)

            # Add edges to the new graph based on the edges in the original graph
            for neighbor in G.neighbors(node):
                neighbor_label = G.nodes[neighbor]["district"]
                if neighbor_label != label:
                    # If the neighbor has a different label, create an edge in the new graph
                    new_G.add_edge(label, neighbor_label)

        return new_G

    def recom_step(self):
        """
        Use recombination MC method to assign new districts
        """

        # Select two adjacent districts
        collapsed_G = self.collapse_graph_by_label(self.graph)
        collapsed_edges = np.array([e for e in collapsed_G.edges])
        random_edge = np.random.choice(np.arange(len(collapsed_edges)))
        district1 = collapsed_edges[random_edge][0]
        district2 = collapsed_edges[random_edge][1]

        total_reps = self.reps[district1 - 1] + self.reps[district2 - 1]
        total_pop = self.pops[district1 - 1] + self.pops[district2 - 1]

        # Combine them into one graph
        g = nx.subgraph(
            self.graph,
            np.array(
                [
                    n
                    for n in self.graph.nodes
                    if self.graph.nodes[n]["district"] in [district1, district2]
                ]
            ),
        )

        # Find the minimal spanning tree
        spanning_tree = nx.algorithms.minimum_spanning_tree(g)
        spanning_tree_edges = np.array([e for e in spanning_tree.edges])

        possible_edge_removals = {}
        # Loop through possible edge cuts
        for i, e in enumerate(spanning_tree_edges):
            possible_edge_removals[i] = 0
            g_ = spanning_tree.copy()
            g_.remove_edge(e[0], e[1])
            reps = 0
            pop = 0
            for j, newdistrict in enumerate(nx.connected_components(g_)):
                pop = np.sum(self.metric[np.array(list(newdistrict))])
                if j == 0:
                    reps = round(
                        np.clip(
                            total_reps * pop / total_pop,
                            a_min=self.min_reps,
                            a_max=min(self.max_reps, total_reps - self.min_reps),
                        )
                    )
                else:
                    reps = total_reps - reps
                possible_edge_removals[i] += np.abs(
                    pop / reps - self.ideal_district_size
                )

        # Pick the best edge to cut
        best_edge = np.argmin(np.array(list(possible_edge_removals.values())))

        # Reassign counties
        spanning_tree.remove_edge(
            spanning_tree_edges[best_edge][0], spanning_tree_edges[best_edge][1]
        )
        components = np.array(list(nx.connected_components(spanning_tree)))
        self.districts[np.array(list(components[0]))] = district1
        self.districts[np.array(list(components[1]))] = district2

    def smart_assign(self, old_flip=None, verbose=False):
        # Loop through all counties
        possible_flips = []
        for county in range(self.n):
            # Identify possible flips by looking at the neighbors of each county
            for neighbor in self.graph.neighbors(county):
                if self.districts[neighbor] != self.districts[county]:
                    G = self.graph.copy()
                    # We can't disconnect any districts
                    # Identify the correct subgraph
                    node_to_district = dict(zip(range(self.n), self.districts))
                    G.remove_node(county)

                    # Find the subgraph of G where the nodes have the same integer as X did
                    subgraph_nodes = [
                        node
                        for node, integer in node_to_district.items()
                        if integer == node_to_district[county]
                    ]
                    subgraph = G.subgraph(subgraph_nodes)

                    if len(subgraph) > 0:
                        if nx.is_connected(subgraph):
                            possible_flips.append(
                                {
                                    "county": county,
                                    "from": self.districts[county],
                                    "to": self.districts[neighbor],
                                }
                            )

        # Loop through possible flips to find the most advantageous one:
        for flip in possible_flips:
            # How many reps do we need to reallocate?
            old_from_reps = self.reps[int(flip["from"]) - 1]
            old_to_reps = self.reps[int(flip["to"]) - 1]
            reps = old_from_reps + old_to_reps

            # Compute new populations of the "from" and "to" districts
            old_from_pop = np.sum(self.metric[np.where(self.districts == flip["from"])])
            old_to_pop = np.sum(self.metric[np.where(self.districts == flip["to"])])

            new_from_pop = old_from_pop - self.metric[flip["county"]]
            new_to_pop = old_to_pop + self.metric[flip["county"]]

            # total population doesn't change, of course
            total_pop = old_from_pop + old_to_pop

            # reallocate representatives for the 2 new districts
            if new_from_pop > new_to_pop:
                new_from_reps = int(
                    np.round(
                        np.clip(
                            reps * new_from_pop / total_pop,
                            a_min=self.min_reps,
                            a_max=min(self.max_reps, reps - self.min_reps),
                        )
                    )
                )
                new_to_reps = reps - new_from_reps
            else:
                new_to_reps = int(
                    np.round(
                        np.clip(
                            reps * new_to_pop / total_pop,
                            a_min=self.min_reps,
                            a_max=min(self.max_reps, reps - self.min_reps),
                        )
                    )
                )
                new_from_reps = reps - new_to_reps

            # compute whether absolute difference from ideal has improved or not
            old_from_diff = np.abs(
                old_from_pop / old_from_reps - self.ideal_district_size
            )
            old_to_diff = np.abs(old_to_pop / old_to_reps - self.ideal_district_size)

            new_from_diff = np.abs(
                new_from_pop / new_from_reps - self.ideal_district_size
            )
            new_to_diff = np.abs(new_to_pop / new_to_reps - self.ideal_district_size)

            flip["diff"] = (old_from_diff + old_to_diff) ** 2 - (
                new_from_diff + new_to_diff
            ) ** 2

        # TODO or make any one district have too many reps

        # make the best flip
        best_flip = np.argmax([f["diff"] for f in possible_flips])

        if old_flip is not None:
            if (
                old_flip["county"] == possible_flips[best_flip]["county"]
                and old_flip["from"] == possible_flips[best_flip]["to"]
            ):
                best_flip = np.argsort([f["diff"] for f in possible_flips])[::-1][
                    np.random.choice(np.arange(1, 10))
                ]

        self.districts[possible_flips[best_flip]["county"]] = possible_flips[best_flip][
            "to"
        ]
        return possible_flips[best_flip]

    def iterate_districts(self):
        """
        Find a candidate swap to do:
        Choose from the most over-represented district, and give to a less-represented neighbor
        (That's the neighborly thing to do!)
        But, go to the next-most over-represented district if there's only 1 county (and so on)
        """

        biggest_districts = np.argsort(self.evaluate_districts())[::-1]
        i = 0
        biggest_district = biggest_districts[i] + 1
        while (
            len(np.where(self.districts == biggest_district)[0]) == 1
            or len(
                np.where(
                    self.representativeness[
                        np.where(self.districts == biggest_district)
                    ]
                    > np.mean(self.representativeness)
                )[0]
            )
            < 2
        ):
            i += 1
            biggest_district = biggest_districts[i] + 1
        old_eval = self.evaluate_districts()

        # Only do this if the donated county is going to be above average in its representativeness
        choice1 = np.random.choice(
            np.where(
                np.logical_and(
                    self.districts == biggest_district,
                    self.representativeness > np.mean(self.representativeness),
                )
            )[0],
            size=1,
        )

        # Find out which (if any) of its neighbors are in a different district
        neighbors = np.where(self.adj_matrix[choice1, :].flatten())[0]
        different_neighbors = neighbors[
            np.where(self.districts[neighbors] != self.districts[choice1])[0]
        ]

        # Assuming there is a neighbor in a different district, find one to donate to
        if len(different_neighbors) > 0:
            choice2 = int(np.random.choice(different_neighbors, size=1))

            # Donate the district - if it doesn't disconnect or destroy the other district
            districts1 = np.where(self.districts == self.districts[choice1])[0]

            # TODO: This won't work if one county is, by itself, bigger than the next-biggest district
            # Need to sort out that edge case.
            if len(districts1) > 1:
                new_districts1 = np.delete(
                    districts1, np.where(districts1 == choice1)[0]
                )
                adj1 = self.adj_matrix[new_districts1][:, new_districts1]
                if nx.is_connected(nx.Graph(adj1)):
                    print(self.districts)
                    print(
                        "Donating county "
                        + str(choice1)
                        + " from district "
                        + str(biggest_district)
                    )
                    print("to district " + str(choice2))
                    print(
                        "District "
                        + str(biggest_district)
                        + " representativeness was "
                        + str(old_eval[int(biggest_district - 1)])
                    )

                    self.districts[choice1] = self.districts[choice2]
                    new_eval = self.evaluate_districts()
                    print(
                        "and is now "
                        + str(new_eval[int(self.districts[biggest_district - 1])])
                    )
                    print(
                        "District "
                        + str(choice2)
                        + " representativeness is now "
                        + str(new_eval[int(self.districts[choice2])])
                    )
                    print("but was " + str(old_eval[int(self.districts[choice2])]))

    def create_districts(self):
        attempts = 0

        while attempts < self.n_attempts and self.num_districts < 50:
            self.initialize_districts()
            results = self.evaluate_districts()
            # flip = self.smart_assign()
            self.recom_step()

            for i in trange(self.n_iterations):
                results = self.evaluate_districts()
                flip = self.smart_assign()
                if self.impossible:
                    # Failure
                    print("Impossible parameter set... perhaps try more districts")
                    attempts = 0
                    self.num_districts += 1
                    break
                if (
                    np.max(results) - np.min(results)
                ) / self.ideal_district_size < self.tolerance:
                    # Success
                    attempts = self.n_attempts
                    self.final_districts = self.districts
                    self.final_num_districts = self.num_districts
                    # self.num_districts += 1
                    print(results)
                    break
            attempts += 1

    def evaluate_districts(self, full_output=False, final=False):
        if not final:
            districts = self.districts
            num_districts = self.num_districts
        else:
            districts = self.final_districts
            num_districts = self.final_num_districts

        metric_df = pd.DataFrame(
            np.concatenate(
                [self.metric.reshape(-1, 1), districts.reshape(-1, 1)], axis=1
            ),
            columns=["metric", "districts"],
        )

        pops = metric_df.groupby("districts").sum(numeric_only=True)["metric"].values

        # Allocate minimum reps to each district
        reps = np.array(self.min_reps + np.zeros((num_districts)), dtype="int")
        remaining_reps = self.n_representatives - np.sum(reps)

        # Allocate remaining reps as best we can
        for i in range(remaining_reps):
            # Identify the district with the most underrepresented people, give them another representative
            # As long as there aren't more than max_reps representatives
            sorted_representation = np.argsort(reps / pops)
            j = 0
            while reps[sorted_representation[j]] >= self.max_reps:
                j += 1
                if j > num_districts - 1:
                    j -= 1
                    self.impossible = True
                    break

            reps[sorted_representation[j]] += 1

        # Store evaluation data
        for i, node in enumerate(self.graph.nodes):
            self.graph.nodes[node]["district"] = int(self.districts[i])

        self.reps = reps
        self.pops = pops

        if full_output:
            self.df["district"] = districts
            dem = self.df.groupby("district").sum(numeric_only=True)["DEM"]
            gop = self.df.groupby("district").sum(numeric_only=True)["GOP"]
            # compute dem and gop leanings of each district
            dem = round(reps * dem / (dem + gop))
            gop = reps - dem
            return np.concatenate(
                [
                    pops.reshape(-1, 1),
                    reps.reshape(-1, 1),
                    pops.reshape(-1, 1) / reps.reshape(-1, 1),
                    dem.values.reshape(-1, 1),
                    gop.values.reshape(-1, 1),
                ],
                axis=1,
            )
        else:
            return pops / reps


data, house2024, senate2024, house_subset, senate_subset = load_data()

# Turn our dataframe into a NetworkX Graph
G = graph_from_df(data)
G = connect_graph(G)

districter = Districter(
    data,
)
