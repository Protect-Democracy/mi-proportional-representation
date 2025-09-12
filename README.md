# Proportional Representation: analysis of Michigan state legislature

Repo for doing analysis of Michigan elections under different proportional electoral systems.

## Quick start
### Prerequisites
- `uv` (follow the instructions [here](https://docs.astral.sh/uv/#installation)). 
### Instructions
Clone the repo & set up Python environment:
```
git clone git@github.com:Protect-Democracy/mi-proportional-representation.git
cd mi-proportional-representation
uv sync
```
Next, you'll need to manually download [this](https://www2.census.gov/geo/tiger/TIGER2024/TABBLOCK20/tl_2024_26_tabblock20.zip) Census block file and extract it into the `raw_data/tiger_blocks` directory. Then you can run the code with:

```
cd src/
uv run redistricting.py
uv run plotting.py
```
When running for the first time, you will have to generate a large number of maps. Depending on how fast your computer is, this may take a while (up to a few hours). 
The maps you generate will be cached locally, so you can make new plots later without having to go through this step again and again.

## What this code does
`redistricting.py` is the meat of the modeling, and does a few things:
- Loads and merges a number of disparate datasets to get a precinct-by-precinct set of election results, where each precinct is also tagged with data points like Census population, number of registered voters, and state House and Senate district.
- Builds a toy model (see footnote) of crossover voting (Democrats voting for Republicans, and vice versa) based on the observed voting patterns in 2024 for President and state legislature.
- Builds a toy model of 4-party crossover voting (in a future world where there are not only a Democratic and Republican party, but, due to proportional representation, also a "progressive" and a "maga" party that are somewhat further to the left and right, respectively.
- Draws a large number (default: 1000) of new legislative maps (to account for the 2030 Census) and estimates the number of Democratic and Republican votes for president in each district. From there, simulates some crossover voting taking place, resulting in legislative seats being awarded to the various parties.
- Applies rules associated with different electoral systems (open list, mixed-member proportional) to the results and tabulates the final mix of parties in the legislature
- Does this in scenarios where Michigan moves to the right, to the left, or stays the same.
- Generates some plots to inspect the results.

Footnote: the "toy models" used here can be found in `redistricting.py` in the `simulate_district_votes` [function](https://github.com/Protect-Democracy/mi-proportional-representation/blob/721932ee1868cc6a54f7d593877a9046a9780d89/src/redistricting.py#L380). 
(This repo remains a work in progress - PRs are welcome!)

## Data
This repository relies on raw data from the following sources:
- `raw_data/tiger_blocks`: TIGER shapefiles containing Census data at the block level in Michigan (needed to ensure that new districts we draw are population-balanced). Download from the Census website [here](https://www2.census.gov/geo/tiger/TIGER2024/TABBLOCK20/tl_2024_26_tabblock20.zip).
- `raw_data/tiger_lower_2024`: TIGER shapefiles for the lower legislative chamber in Michigan (i.e., the House of Representatives). Available from the Census [website](https://www.census.gov/cgi-bin/geo/shapefiles/index.php?year=2024&layergroup=State+Legislative+Districts).
- `raw_data/tiger_upper_2024`: TIGER shapefiles for the lower legislative chamber in Michigan (i.e., the Senate). Available from the Census [website](https://www.census.gov/cgi-bin/geo/shapefiles/index.php?year=2024&layergroup=State+Legislative+Districts).
- `2024GEN`: Raw 2024 general election results in Michigan. Available from the Michigan Secretary of State [website](https://mvic.sos.state.mi.us/votehistory/). Note that the web interface only offers county-by-county views of the data, but clicking the "Download Precinct Results" button gives the data for the entire state.
- `2024_Voting_Precincts`: Shapefile of voting precincts in Michigan for the 2024 general election. Available for download from the Michigan ArcGIS [site](https://gis-michigan.opendata.arcgis.com/datasets/2024-voting-precincts/explore).
- `detroit_crosswalk.csv`: Detroit has about 400 physical precincts, and about 65 absentee ballot counting boards that are counted separately in the results. Because we want to draw new geographic boundaries, we need a way to map the 65 counting boards to the precincts they represent, which is what this crosswalk file does. Source: private correspondence with Michigan Department of State.
- `plural.csv`: A CSV of the current makeup of the Michigan state legislature. Not used in the current code, but useful for determining who won each district and computing the current partisan balance of the legislature. From Plural's [openstate](https://open.pluralpolicy.com/data/legislator-csv/) data page. 


## Example Plots
2024 2-party presidential vote share (red = Republican, blue = Democratic) by precinct:
![party_share_plot](https://github.com/user-attachments/assets/1bf738fc-f7f6-4b9f-b549-8232258110fe)
