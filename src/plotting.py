"""
Plotting code - run this file to produce the summary
plots in the ../plots/ directory
"""

import pickle
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from scipy.stats import binned_statistic, norm


from redistricting import (
    get_district_maps,
    run_election,
    load_and_prepare_graph,
    simulate_district_votes,
    fit_statistical_models,
)
from functools import wraps
import os


# Consistent use of colors for parties
MAGA_COLOR = "orange"
GOP_COLOR = "red"
DEM_COLOR = "blue"
PROG_COLOR = "green"

# Setting some rcParams for consistent plot styling
plt.rcParams["text.color"] = "#2E2D31"
plt.rcParams["axes.titlecolor"] = "#2E2D31"
plt.rcParams["axes.labelcolor"] = "#2E2D31"
plt.rcParams["axes.edgecolor"] = "#2E2D31"
plt.rcParams["xtick.color"] = "#2E2D31"
plt.rcParams["ytick.color"] = "#2E2D31"
plt.rcParams["xtick.labelcolor"] = "#2E2D31"
plt.rcParams["ytick.labelcolor"] = "#2E2D31"
plt.rcParams["savefig.facecolor"] = "#FBF6ED"
plt.rcParams["figure.facecolor"] = "#FBF6ED"
plt.rcParams["axes.facecolor"] = "#FBF6ED"

plt.rcParams["savefig.facecolor"] = "gray"
plt.rcParams["figure.facecolor"] = "gray"
plt.rcParams["axes.facecolor"] = "gray"


def plot_styler(
    save_dir="../plots/",
    facecolor="#FBF6ED",
):
    """
    A decorator for matplotlib plotting functions to abstract away boilerplate code.

    This decorator handles:
    1. Setting the background colors for the figure, axes, and saved file.
    2. Saving the figure to a specified directory if a 'filename' kwarg is provided.
    3. Showing the plot (can be disabled with show_plot=False).
    4. Cleaning up the plot figure after saving/showing to prevent state leakage.

    Args:
        save_dir (str): The directory where plots will be saved.
        savefig_facecolor (str): Background color for the saved image.
        figure_facecolor (str): Background color for the figure.
        axes_facecolor (str): Background color for the axes.
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            """
            Wrapper function that applies styling, saves, and shows the plot.

            NOTE: The decorated function should not call plt.savefig() or plt.show().

            Additional kwargs accepted by the wrapper at call time:
                filename (str, optional): If provided, the plot is saved with this
                                          name in the `save_dir`. Defaults to None.
                show_plot (bool, optional): If True, plt.show() is called.
                                            Defaults to True.
            """
            # Store original rcParams to restore them later
            original_params = {
                "savefig.facecolor": plt.rcParams["savefig.facecolor"],
                "figure.facecolor": plt.rcParams["figure.facecolor"],
                "axes.facecolor": plt.rcParams["axes.facecolor"],
            }

            try:
                # Apply new styles from the decorator arguments
                plt.rcParams["savefig.facecolor"] = facecolor
                plt.rcParams["figure.facecolor"] = facecolor
                plt.rcParams["axes.facecolor"] = facecolor

                # Call the user's plotting function
                # This function is expected to create a plot but not show/save it
                func_result = func(*args, **kwargs)

                filename = kwargs.get("filename")
                show_plot = kwargs.get("show_plot", True)

                plt.tight_layout()

                # Save the figure if a filename is provided
                if filename:
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir)
                    full_path = os.path.join(save_dir, filename)
                    plt.savefig(full_path)
                    print(f"Plot saved to '{full_path}'")

                # Show the plot if requested
                if show_plot:
                    plt.show()

                # Clean up the current figure to prevent it from affecting subsequent plots
                plt.close("all")

                return func_result

            finally:
                # Restore original rcParams to avoid side effects on other plots
                plt.rcParams.update(original_params)

        return wrapper

    return decorator


@plot_styler(facecolor="gray")
def pop_density_map(blocks, filename=None):
    """Look at Census data on population density."""
    df = blocks.copy()
    df["popdensity"] = df["POP20"] / df["ALAND20"]

    plt.figure(figsize=[15, 15])
    df.plot(
        "popdensity",
        cmap="plasma",
        edgecolor="None",
        linewidth=0,
        vmin=0,
        vmax=2 * 1e-3,
        ax=plt.gca(),
    )
    plt.gca().set_axis_off()


@plot_styler(facecolor="gray")
def party_share_plot(data, filename=None):
    """Look at 2024 2-party vote share (red = GOP, blue = Dem)."""
    df = data.copy()
    df["2partyshare"] = df["DONALD J. TRUMP"] / (
        df["DONALD J. TRUMP"] + df["KAMALA D. HARRIS"]
    )
    plt.figure(figsize=[15, 15])
    df.plot("2partyshare", ax=plt.gca(), cmap="coolwarm", vmin=0, vmax=1)

    plt.gca().set_axis_off()


@plot_styler(facecolor="lightgray")
def district_heatmap(precincts, districts, filename=None):
    """Same as the party_share_plot, but overlaid with district boundaries."""
    plt.figure(figsize=[20, 20])

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
    if districts is not None:
        districts.plot(edgecolor="k", linewidth=2.0, ax=plt.gca(), facecolor="None")
    plt.gca().set_axis_off()


@plot_styler(facecolor="lightgray")
def crossover_plot(data, models, filename=None):
    plt.figure(figsize=[12, 6])
    plt.subplot(121)

    plt.scatter(
        data["KAMALA D. HARRIS"] / (data["KAMALA D. HARRIS"] + data["DONALD J. TRUMP"]),
        data["STATE_REP_DEM"] / (data["STATE_REP_DEM"] + data["STATE_REP_GOP"]),
        alpha=0.25,
        s=10,
        label="Precincts",
    )
    plt.xlabel("D Vote Fraction (POTUS)")
    plt.ylabel("D Vote Fraction (State Legislature)")
    plt.legend(loc="lower right")
    plt.subplot(122)

    h = np.histogram(
        data["KAMALA D. HARRIS"] / (data["KAMALA D. HARRIS"] + data["DONALD J. TRUMP"])
        - data["STATE_REP_DEM"] / (data["STATE_REP_DEM"] + data["STATE_REP_GOP"]),
        bins=np.linspace(-0.5, 0.5, 201),
        density=True,
    )
    plt.hist(
        data["KAMALA D. HARRIS"] / (data["KAMALA D. HARRIS"] + data["DONALD J. TRUMP"])
        - data["STATE_REP_DEM"] / (data["STATE_REP_DEM"] + data["STATE_REP_GOP"]),
        bins=np.linspace(-0.5, 0.5, 201),
        density=True,
        label="Data",
    )
    plt.xlabel("Vote Difference")
    plt.ylabel("Frequency")
    # plt.plot(
    #     h[1][:-1], norm.pdf(h[1][:-1], loc=res.x[0], scale=res.x[1]), label="Model fit"
    # )
    plt.legend(loc="upper right")


@plot_styler()
def legislature_histogram(
    all_results,
    scenario_name="OL9",
    partisan_change="no_change",
    n_parties=4,
    filename=None,
):
    fig = plt.figure(figsize=[10, 6])
    bins = np.arange(100)
    if n_parties == 4:
        plt.hist(
            [v["P"] for v in all_results[scenario_name][partisan_change][n_parties]],
            alpha=0.5,
            bins=bins,
            label="Progressive",
            facecolor=PROG_COLOR,
        )
        plt.hist(
            [v["M"] for v in all_results[scenario_name][partisan_change][n_parties]],
            alpha=0.5,
            bins=bins,
            label="MAGA",
            facecolor=MAGA_COLOR,
        )
    else:
        plt.axvline(19 + 64 * (100 / 110), ls="--", lw=2.0, c="red")
        plt.axvline(19 + 46 * (100 / 110), ls="--", lw=2.0, c="blue")
    plt.hist(
        [v["D"] for v in all_results[scenario_name][partisan_change][n_parties]],
        alpha=0.5,
        bins=bins,
        label="DEM",
        facecolor=DEM_COLOR,
    )
    plt.hist(
        [v["R"] for v in all_results[scenario_name][partisan_change][n_parties]],
        alpha=0.5,
        bins=bins,
        label="GOP",
        facecolor=GOP_COLOR,
    )
    plt.legend(loc="upper left")
    plt.xlabel("Seats")
    plt.ylabel("Frequency")


@plot_styler()
def precinct_partisanship_plot(districts, filename=None):
    fig = plt.figure(figsize=[10, 6])
    bins = np.linspace(0, 1, 71)
    plt.hist(districts["2partyshare"], color="darkgray", bins=bins, density=True)
    plt.hist(
        districts["2partyshare"],
        histtype="step",
        bins=bins,
        color="k",
        density=True,
        lw=1.5,
    )
    plt.ylabel("Frequency")
    plt.xlabel("Precinct partisanship")


@plot_styler()
def stacked_precinct_model_plot(districts, models, filename=None):
    fig = plt.figure(figsize=[10, 6])
    bins = np.linspace(0, 1.0, 71)
    s = simulate_district_votes(
        pd.DataFrame({"2partyshare": bins[:-1]}),
        models,
        4,
        "no_change",
        crossover_method=None,
    )
    h = np.histogram(districts["2partyshare"], bins=bins, density=True)
    plt.hist(
        districts["2partyshare"],
        histtype="step",
        bins=bins,
        color="k",
        density=True,
        lw=1.5,
    )

    labels = ["Progressive", "Democratic", "Republican", "MAGA"]
    colors = [PROG_COLOR, DEM_COLOR, GOP_COLOR, MAGA_COLOR]
    y_offset = np.zeros_like(s[:, 0] * h[0])
    legend_handles = []

    for i, (label, color) in enumerate(zip(labels, colors)):
        y_data = s[:, i] * h[0]
        plt.fill_between(
            bins[:-1], y_offset, y_offset + y_data, step="post", color=color
        )
        y_offset += y_data
        plt.step(bins[:-1], y_offset, where="post", color="k", lw=1.0)
        legend_handles.append(
            mpatches.Patch(facecolor=color, edgecolor="k", label=label)
        )

    plt.xlim([0, 1])
    plt.legend(handles=legend_handles, loc="upper right")
    plt.ylabel("Frequency")
    plt.xlabel("Precinct partisanship")


@plot_styler(facecolor="gray")
def district_crossover_heatmap(precincts, districts=None, filename=None):
    fig = plt.figure(figsize=[20, 20])

    df = precincts.copy()
    df["2potusshare"] = df["DONALD J. TRUMP"] / (
        df["DONALD J. TRUMP"] + df["KAMALA D. HARRIS"]
    )
    df["2partyshare"] = df["STATE_REP_GOP"] / (
        df["STATE_REP_GOP"] + df["STATE_REP_DEM"]
    )

    df["plot"] = df["2potusshare"] - df["2partyshare"]
    df.plot(
        "plot",
        ax=plt.gca(),
        cmap="Spectral_r",
        vmin=-0.15,
        vmax=0.15,
        linewidth=0.0,
        edgecolor="None",
    )

    if districts is not None:
        districts.plot(edgecolor="k", linewidth=2.0, ax=plt.gca(), facecolor="None")
    plt.gca().set_axis_off()


def hypothetical_legislature_from_potus(data):
    # What would the legislature have looked like if House seats were
    # determined by POTUS vote in 2024?
    current_grouped = data.groupby("SLDLST")[
        ["KAMALA D. HARRIS", "DONALD J. TRUMP", "STATE_REP_GOP", "STATE_REP_DEM"]
    ].sum()
    current_grouped["2potusshare"] = current_grouped["DONALD J. TRUMP"] / (
        current_grouped["DONALD J. TRUMP"] + current_grouped["KAMALA D. HARRIS"]
    )
    current_grouped["2partyshare"] = current_grouped["STATE_REP_GOP"] / (
        current_grouped["STATE_REP_GOP"] + current_grouped["STATE_REP_DEM"]
    )
    current_grouped["R_share"] = 1.0 * (
        current_grouped["STATE_REP_GOP"] > current_grouped["STATE_REP_DEM"]
    )
    current_grouped["R_share_potus"] = 1.0 * (current_grouped["2potusshare"] > 0.5)
    current_rs = int(np.sum(current_grouped["R_share"]))
    hypothetical_rs = int(np.sum(current_grouped["R_share_potus"]))
    print(f"Actual House composition: {current_rs} R, {110 - current_rs} D")
    print(
        f"POTUS-based House composition: {hypothetical_rs} R, {110 - hypothetical_rs} D"
    )


def main():
    graph, data, house2024, senate2024 = load_and_prepare_graph()
    data["2partyshare"] = data["DONALD J. TRUMP"] / (
        data["KAMALA D. HARRIS"] + data["DONALD J. TRUMP"]
    )
    models = fit_statistical_models(data)

    # Census data map
    pop_density_map(data, filename="population_density.png")

    # Vote share maps
    party_share_plot(data, filename="party_share_plot.png")
    district_heatmap(data, districts=house2024, filename="house_districts.png")
    district_heatmap(data, districts=senate2024, filename="senate_districts.png")

    # Plots to describe the 4-party model
    precinct_partisanship_plot(data, filename="precinct_partisanship.png")
    stacked_precinct_model_plot(data, models, filename="stacked_precinct_model.png")

    # where did the state representative outperform the POTUS candidate?
    district_crossover_heatmap(data, filename="relative_performance.png")

    # a quick analysis function that is interesting
    hypothetical_legislature_from_potus(data)

    # Plots to evaluate the redistricting simulation results
    try:
        with open("simulation_results.pkl", "rb") as f:
            all_results = pickle.load(f)
    except:
        print("Simulation results not found! Please run redistricting.py first.")

    # Examples of examining legislature results
    legislature_histogram(
        all_results, scenario_name="OL5", filename="OL5_4_parties_no_change.png"
    )
    legislature_histogram(
        all_results, scenario_name="OL9", filename="OL9_4_parties_no_change.png"
    )
    legislature_histogram(
        all_results, scenario_name="OL1", filename="OL1_4_parties_no_change.png"
    )
    legislature_histogram(all_results, scenario_name="MMP")
    legislature_histogram(all_results, scenario_name="OL1", n_parties=2)
    legislature_histogram(all_results, scenario_name="OL5", n_parties=2)
    legislature_histogram(all_results, scenario_name="OL9", n_parties=2)
    legislature_histogram(all_results, scenario_name="MMP", n_parties=2)


if __name__ == "__main__":
    main()
