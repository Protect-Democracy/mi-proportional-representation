import geopandas as gp
from gerrychain import Graph, Partition
from gerrychain.updaters import Tally, cut_edges
from libpysal.weights import Queen
import networkx as nx
from utils import dissolve_small_into_large
from data_loading import (
    load_and_format_precincts_shapefile,
    load_legislature,
    load_and_format_votes,
    reallocate_detroit_counting_board_votes,
    merge_shapefiles_and_votes,
)


def draw_new_districts():
    """Under construction!!"""

    w = Queen.from_dataframe(data, idVariable="unique_precinct")
    G = nx.from_dict_of_lists(w.neighbors)

    attributes_dict = {
        row["unique_precinct"]: row.to_dict() for index, row in data.iterrows()
    }
    # 2. Set the node attributes in the graph
    nx.set_node_attributes(G, attributes_dict)

    for node in G:
        print(G.nodes[node]["unique_precinct"])
        break

    g = Graph.from_networkx(G)
    initial_partition = Partition(
        g,
        assignment="unique_precinct",
        updaters={
            "population": Tally("POP20", alias="population"),
            "cut_edges": cut_edges,
        },
    )

    for district, pop in initial_partition["population"].items():
        print(f"District {district}: {pop}")


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
