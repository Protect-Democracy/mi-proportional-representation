import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.cm import get_cmap
from scipy.stats import norm
from scipy.stats import gaussian_kde

from data_loading import (
    load_data,
)
from redistricting import (
    two_component_gaussian,
    four_component_gaussian,
    fit_crossover,
    fit_shor_mccarty,
    party_popularity,
)


def district_heatmap(precincts, districts, filename=None):
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
    if filename is not None:
        plt.savefig("plots/" + filename)
    plt.show()


def partisanship_scatterplot(house_subset, senate_subset, filename=None):
    plt.scatter(house_subset["np_score"], house_subset["2partyshare"], label="house")
    plt.scatter(senate_subset["np_score"], senate_subset["2partyshare"], label="senate")
    plt.xlabel("Shor-McCarty score")
    plt.ylabel("2024 2-party share")
    plt.legend()
    if filename is not None:
        plt.savefig("plots/" + filename)
    plt.show()


def pop_density_map(blocks, filename=None):
    df = blocks.copy()
    df["popdensity"] = df["POP20"] / df["ALAND20"]
    fig = plt.figure(figsize=[20, 20])
    df.plot(
        "popdensity", cmap="YlGnBu", edgecolor="None", linewidth=0, vmin=0, vmax=1e-3
    )
    plt.gca().set_axis_off()
    plt.tight_layout()
    if filename is not None:
        plt.savefig("plots/" + filename)
    plt.show()


def party_share_plot(data, filename=None):
    df = data.copy()
    df["2partyshare"] = df["DONALD J. TRUMP"] / (
        df["DONALD J. TRUMP"] + df["KAMALA D. HARRIS"]
    )
    fig = plt.figure(figsize=[15, 15])
    df.plot("2partyshare", ax=plt.gca(), cmap="coolwarm", vmin=0, vmax=1)

    plt.gca().set_axis_off()
    plt.tight_layout()
    if filename is not None:
        plt.savefig("plots/" + filename)
    plt.show()


def shor_mccarty_plots(leg):
    cmap = get_cmap("coolwarm")
    for year in range(2000, 2023):
        year_col = [c for c in leg.columns if str(year) in c][0]
        subleg = leg[~leg[year_col].isna()]
        plt.hist(
            subleg["np_score"],
            bins=np.linspace(-2.0, 2.0, 101),
            histtype="step",
            color=cmap((year - 2000) / 22),
        )
    plt.show()

    year_range = range(1996, 2023)
    bins = np.linspace(-2, 2, 101)
    M = np.zeros((len(year_range), len(bins) - 1))
    for i, year in enumerate(year_range):
        year_col = [c for c in leg.columns if str(year) in c][0]
        subleg = leg[~leg[year_col].isna()]
        M[i] = np.histogram(subleg["np_score"], bins=bins, density=True)[0]
    plt.imshow(
        M.T, origin="lower", aspect=7, extent=[year_range[0], year_range[-1], -2, 2]
    )
    plt.ylabel("Shor-McCarty Score")
    plt.show()


def crossover_plot(data, res, filename=None):
    fig = plt.figure(figsize=[12, 6])
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
    plt.plot(
        h[1][:-1], norm.pdf(h[1][:-1], loc=res.x[0], scale=res.x[1]), label="Model fit"
    )
    plt.legend(loc="upper right")
    plt.tight_layout()

    if filename is not None:
        plt.savefig("plots/" + filename)
    plt.show()


def party_breakdown_plot(leg, res, result2, result4, filename=None):
    fig = plt.figure(figsize=[12, 5])
    plt.subplot(121)
    xspace = np.linspace(-2, 2, 101)
    plt.hist(
        leg["np_score"],
        bins=np.linspace(-2, 2, 101),
        density=True,
        color="tab:blue",
        label="Data",
    )
    plt.plot(
        xspace,
        two_component_gaussian(result2.x, xspace),
        c="w",
        lw=2,
        ls="--",
        label="2 component fit",
    )
    plt.plot(
        xspace,
        four_component_gaussian(result4.x, xspace),
        c="violet",
        lw=2,
        ls="--",
        label="4 component fit",
    )
    plt.legend(loc="upper left")
    plt.xlabel("Shor-McCarty Index")
    plt.ylabel("Frequency")
    plt.title("All US Legislators")
    plt.xlim([-2, 2])
    plt.subplot(122)
    plt.plot(
        xspace,
        result4.x[0] * norm(loc=result4.x[1], scale=result4.x[2]).pdf(xspace),
        c="green",
        lw=2,
        ls="--",
        label="Progressive Party",
    )
    plt.plot(
        xspace,
        result4.x[3] * norm(loc=result4.x[4], scale=result4.x[5]).pdf(xspace),
        c="blue",
        lw=2,
        ls="--",
        label="Democratic Party",
    )
    plt.plot(
        xspace,
        result4.x[6] * norm(loc=result4.x[7], scale=result4.x[8]).pdf(xspace),
        c="orange",
        lw=2,
        ls="--",
        label="Republican Party",
    )
    plt.plot(
        xspace,
        result4.x[9] * norm(loc=result4.x[10], scale=result4.x[11]).pdf(xspace),
        c="r",
        lw=2,
        ls="--",
        label="MAGA Party",
    )
    plt.plot(
        xspace,
        two_component_gaussian(result2.x, xspace),
        c="w",
        lw=2,
        ls="--",
        label="2 component fit",
    )
    plt.ylim([0, 0.7])
    plt.xlim([-2, 2])
    plt.xlabel("Shor-McCarty Index")
    plt.ylabel("Frequency")
    plt.legend(loc="upper left")
    plt.title("4-Party Model")
    plt.tight_layout()
    if filename is not None:
        plt.savefig("plots/" + filename)
    plt.show()


def crossover_voting_plots(house_subset, senate_subset, filename=None):
    """Under construction"""
    # for some reason, mapping partisan leaning onto our party model
    # seems to do a good job in a 2-party system of predicting the winner
    # Now see if this model matches the observed data
    kde = gaussian_kde(
        np.vstack(
            [
                np.hstack([senate_subset["2partyshare"], house_subset["2partyshare"]]),
                np.hstack([senate_subset["np_score"], house_subset["np_score"]]),
            ]
        ),
        bw_method=0.25,
    )
    bins = np.linspace(0, 1, 51)
    dem_prob = np.zeros((50))
    gop_prob = np.zeros((50))
    for i in range(len(bins) - 1):
        dem_prob[i] = kde.integrate_box([bins[i], -2.0], [bins[i + 1], 0.0])
        gop_prob[i] = kde.integrate_box([bins[i], 0.0], [bins[i + 1], 2.0])
        dem_prob[i] = dem_prob[i] / (dem_prob[i] + gop_prob[i])
        gop_prob[i] = 1.0 - dem_prob[i]
    w = np.zeros((len(bins)))
    for i, bin in enumerate(bins):
        y, z = party_popularity(bin, result4)
        w[i] = y[0] / (y[1] + y[0])
    plt.fill_between(bins[:-1], 0, gop_prob, color="r")
    plt.fill_between(bins[:-1], gop_prob, 1.0, color="b")
    plt.plot(bins, 1.0 - w, ls="--", c="k", lw=2.0)
    plt.xlim([0, 1])
    plt.ylim([0, 1])

    plt.tight_layout()
    if filename is not None:
        plt.savefig("plots/" + filename)
    plt.show()


def summary_plots(legislature_dict, filename=None):
    """Under construction"""
    for scenario in legislature_dict.keys():
        for n_parties in [2, 4]:
            leg_df = pd.DataFrame(legislature_dict[scenario]["no_change"][n_parties])
            plt.hist(leg_df["D"], bins=range(0, 100), label="Democrat", alpha=0.7)
            plt.hist(leg_df["R"], bins=range(0, 100), label="Republican", alpha=0.7)
            if n_parties > 2:
                plt.hist(leg_df["M"], bins=range(0, 100), label="MAGA", alpha=0.7)
                plt.hist(
                    leg_df["P"], bins=range(0, 100), label="Progressive", alpha=0.7
                )
            plt.ylabel("Count")
            plt.xlabel("n_parties in legislature")
            plt.title(scenario + ", " + str(n_parties) + "Parties")
            plt.legend()
            plt.show()


if __name__ == "__main__":
    # Load voting results
    data, house2024, senate2024, house_subset, senate_subset = load_data()

    party_share_plot(data, filename="party_share_plot.pdf")
    district_heatmap(data, house2024)
    district_heatmap(data, senate2024)
    partisanship_scatterplot(house_subset, senate_subset)

    leg = pd.read_stata("raw_data/legislator_data/shor_mccarty.dta")
    shor_mccarty_plots(leg)

    res = fit_crossover(data)
    result2, result4 = fit_shor_mccarty()
    crossover_plot(data, res)
    party_breakdown_plot(leg, res, result2, result4)
