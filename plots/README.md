# A quick overview of the results of the simulation

## Election results
Let's start by looking at `party_share_plot.png`, which shows the 2024 2-party vote share for President (blue = more Democratic, red = more Republican):
![plots/party_share_plot.png](https://github.com/Protect-Democracy/mi-proportional-representation/blob/8d997f59172123c62163149010f48738cddcda2f/plots/party_share_plot.png)

Note that most of the map is red, and the blue is concentrated in the cities. We can bin this data up to view this bimodal distribution:
![plots/precinct_partisanship.png](https://github.com/Protect-Democracy/mi-proportional-representation/blob/8d997f59172123c62163149010f48738cddcda2f/plots/precinct_partisanship.png)

This confirms what we could have guessed: Democrats tend to be clustered in cities (the sharp spike on the left), whereas Republicans tend to be more common in exurban and rural areas.

Under a proportional representation system, we would expect minor parties to form more readily than under the current winner-take-all system, because the threshold for winning seats is lower. 
However, we can only guess at what such a system would look like.
In this case, we'll guess that there would be 4 parties in a proportional system: a far-left "Progressive" party, a center-left "Democratic" party, a center-right "Republican" party, and a far-right "MAGA" party.

How can we extrapolate from 2024 data to such a system? Again, we have to make some modeling decisions. In this case, we assumed that the existing Democratic and Republican parties would split in two, into the Progressive/Democratic and Republican/MAGA parties, respectively.
We also made some assumptions that because of network effects, the Progressive party would be somewhat more prevalent in the areas where Democrats are common today, and the MAGA party would be more prevalent in areas where Republicans are more common today.

You can see what this looks like in `stacked_precinct_model.png`:
![plots/stacked_precinct_model.png](https://github.com/Protect-Democracy/mi-proportional-representation/blob/8d997f59172123c62163149010f48738cddcda2f/plots/stacked_precinct_model.png)

With our model in hand, we can draw new population-balanced districts, sum up the number of votes in each for each district (applying some random ticket-splitting along the way, to account for varying candidate strength), and compute the legislative makeup.
By drawing a large number of maps, we can get a sense of what is reasonable to expect under our model. 
Legislatures can be viewed in plots like `OL5_4_parties_no_change.png` (no_change refers to the idea that the presidential vote in the future is similar to the 2024 vote; we also simulate scenarios where the state drifts further right or left):
![plots/OL5_4_parties_no_change.png](https://github.com/Protect-Democracy/mi-proportional-representation/blob/8d997f59172123c62163149010f48738cddcda2f/plots/OL5_4_parties_no_change.png)

Here we see that the far-left and far-right parties gain a considerable number of seats, but they each remain in the minority (as expected under a proportional system). 

You can generate your own plots by modifying the code in [src/plotting.py](https://github.com/Protect-Democracy/mi-proportional-representation/blob/8d997f59172123c62163149010f48738cddcda2f/src/plotting.py). 
