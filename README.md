# mi-proportional-representation

Repo for doing analysis of Michigan elections under different proportional electoral systems.

## What this code does
`redistricting.py` is the meat of the modeling, and does a few things:
- Loads and merges a number of disparate datasets to get a precinct-by-precinct set of election results, where each precinct is also tagged with data points like Census population, number of registered voters, and state House and Senate district.
- Builds a toy model of crossover voting (Democrats voting for Republicans, and vice versa) based on the observed voting patterns in 2024 for President and state legislature.
- Builds a toy model of 4-party crossover voting (in a future world where there are not only a Democratic and Republican party, but, due to proportional representation, also a "progressive" and a "maga" party that are somewhat further to the left and right, respectively.
- Draws a large number of new legislative maps (to account for the 2030 Census) and estimates the number of Democratic and Republican votes for president in each district. From there, simulates some crossover voting taking place, resulting in legislative seats being awarded to the various parties.
- Applies rules associated with different electoral systems (open list, mixed-member proportional) to the results and tabulates the final mix of parties in the legislature
- Does this in scenarios where Michigan moves to the right, to the left, or stays the same.

(This repo remains a work in progress - PRs are welcome!)

## Usage

Clone this repo and then run one of the scripts in the `src/` directory. For example, to get a bunch of plots:
`uv run src/plotting.py`

When running for the first time, you may have to download some Census data to your local machine, which will take a few moments.

## Example Plots
2024 2-party presidential vote share (red = Republican, blue = Democratic) by precinct:
![party_share_plot](https://github.com/user-attachments/assets/1bf738fc-f7f6-4b9f-b549-8232258110fe)
