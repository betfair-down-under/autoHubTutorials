# %% [markdown]
# # Analysing and Predicting The BSP
# 
# 
# ## 0.1 Setup
# 
# Once again I'll be presenting the analysis in a jupyter notebook and will be using python as a programming language.
# 
# Some of the data processing code takes a while to execute - that code will be in cells that are commented out - and will require a bit of adjustment to point to places on your computer locally where you want to store the intermediate data files.
# 
# You'll also need `betfairlightweight` which you can install with something like `pip install betfairlightweight`.

# %%
import pandas as pd
import numpy as np
import requests
import os
import re
import csv
import plotly.express as px
import plotly.graph_objects as go
import math
import logging
import yaml
import csv
import tarfile
import zipfile
import bz2
import glob
import ast

from datetime import date, timedelta
from unittest.mock import patch
from typing import List, Set, Dict, Tuple, Optional
from itertools import zip_longest
import betfairlightweight
from betfairlightweight import StreamListener
from betfairlightweight.resources.bettingresources import (
    PriceSize,
    MarketBook
)

# %% [markdown]
# ## 0.2 Context
# 
# The BSP is betting product offered by betfair (on large enough markets) that gives customers a chance to back or lay any selection at a "fair" price. Without getting too complex too quickly, the BSP allows you lock in a bet at any time after the market is openened and for as much stake as you can afford. The BSP is a good option for many different segments of customers:
# 
# - Recreational punters that don't have a particular strategy for trying to get the best odds can lock in a price that is (in the aggregate) a lot better than what they'd get at a corporate book or they'd get by taking limit bets early in a market's trading
# - Automated customers that don't want the hassle of managing live market trading can implement automated strategies a lot easier whilst also protecting them from edge cases like race reschedules 
# - Is perfect for simply backtesting fundemental models as it's a resiliant and robust single price
# 
# Despite it being a good option for a lot of customers it's also a fairly contraversial topic for some other types of customers. Some people firmly believe that the BSP on big markets reflects the "true chance" of a selection so betting it is a fools errand that will simply lose you commission over the long run. You might have heard a version of this story before: given the large pool sizes, the 0% overround, the settlement at the exact moment the market is suspended the BSP perfectly synthesises all available public information and demand and arrives at a true fair odds. Some will attempt to prove this to you by showing you a predicted chance vs observed win rate scatterplot which shows a perfect correlation between chance implied by the BSP and a horses true chance. Whilst I don't disagree that the BSP is a **very strong** estimate of a selections chance it's pretty obviously not perfect. 
# 
# Furthermore, it presents some other tricky challenges to use in practical situations. It's not knowable perfectly before it's the exact moment of market suspension so many model or strategy builders make the mistake of unknowingly leaking it into their preplay model development or their theoretical staking calculations. Where the final number will land is actually another source of uncertainty in your processes which presents anothing forecasting / predictive modelling application as I'll explore later in this piece. I'll take you through how I'd measure the accuracy of the BSP, show you how it's traded on the exchange, and take you through a host of methods of estimating the BSP and build a custom machine learning approach that's better than each of them.
# 
# ## 0.3 The Algorithm
# 
# The actual logic of how betfair arrives at the final BSP number is quite complex and for a few reasons you won't be able to perfectly replicate it at home. However, the general gist of the BSP reconciliation algorithm that is executed just as the market suspended goes something like:
# 
# - The algorithm combines 4 distinct groups of open bets for a given selection: 
#     + Non price limited BSP orders on both the back and lay side (`market_on_close` orders)
#     + Price limited orders on both the back and lay side (`limit_on_close` orders)
#     + All non filled open lay orders
#     + All non filled open back orders 
# - It then combines them all together, passes a sophisticated balancing algorithm over the top of them and arrives at a single fair price for the BSP that balances the demand on either side of the ledger
# 
# ## 0.4 This Example
# 
# For this exercise we'll again take advantage of the betfair historical stream json files. The slice of betfair markets I'll be analysing is all thoroughbred races over July 2020 - June 2021. 
# 
# As an aside the projected BSP number you see on the betfair website isn't collected inside betfair's own internal database of orders, so any custom data request you may be able to get as a VIP won't include this number. So if you were planning to include it in any kind of projection or bet placement logic operation you were making the only way to anlayse it historically is to mine these data files. Another good reason to learn the skills to do so! 

# %% [markdown]
# # 1.0 Data
# 
# Like the previous tutorial we won't be able to collapse the stream data down into a single row per runner because I'm interested in anlaysing how the projected BSP moves late in betfair markets. I'm also interested in plotting the efficiency of certain odds values at certain distinct time points leading up the the races so I need multiple records per runner.
# 
# Like in the previous tutorial I'll split out the selection metadata, BSP and win flag values as a seperate data file to reduce the size of the datafiles extracted for this analysis.
# 
# For the preplay prices dataset I'll:
# 
# - Start extraction at 2 mins before the scheduled off
# - Extract prices every 10 seconds thereafter until the market is suspended
# - I'll also extract the final market state the instant before the market is suspended
# 
# ## 1.1 Sourcing Data
# 
# First you'll need to source the stream file TAR archive files. I'll be analysing 12 months of Australian thoroughbred Pro files. Aask automation@betfair.com.au for more info if you don't know how to do this. Once you've gotten access download them to your computer and store them together in a folder.
# 
# ## 1.2 Utility functions
# 
# First like always we'll need some general utility functions that you may have seen before:

# %%
# General Utility Functions
# _________________________________

def split_anz_horse_market_name(market_name: str) -> (str, str, str):
    parts = market_name.split(' ')
    race_no = parts[0] # return example R6
    race_len = parts[1] # return example 1400m
    race_type = parts[2].lower() # return example grp1, trot, pace
    return (race_no, race_len, race_type)


def load_markets(file_paths):
    for file_path in file_paths:
        print(file_path)
        if os.path.isdir(file_path):
            for path in glob.iglob(file_path + '**/**/*.bz2', recursive=True):
                f = bz2.BZ2File(path, 'rb')
                yield f
                f.close()
        elif os.path.isfile(file_path):
            ext = os.path.splitext(file_path)[1]
            # iterate through a tar archive
            if ext == '.tar':
                with tarfile.TarFile(file_path) as archive:
                    for file in archive:
                        yield bz2.open(archive.extractfile(file))
            # or a zip archive
            elif ext == '.zip':
                with zipfile.ZipFile(file_path) as archive:
                    for file in archive.namelist():
                        yield bz2.open(archive.open(file))

    return None

def pull_ladder(availableLadder, n = 5):
        out = {}
        price = []
        volume = []
        if len(availableLadder) == 0:
            return(out)        
        else:
            for rung in availableLadder[0:n]:
                price.append(rung.price)
                volume.append(rung.size)

            out["p"] = price
            out["v"] = volume
            return(out)

def filter_market(market: MarketBook) -> bool: 
    
    d = market.market_definition

    return (d.country_code == 'AU' 
        and d.market_type == 'WIN' 
        and (c := split_anz_horse_market_name(d.name)[2]) != 'trot' and c != 'pace')


# %% [markdown]
# ## 1.3 Selection Metadata
# 
# Given that the detailed price data will have so many records we will split out the selection metadata (including the selection win outcome flag and the bsp) into it's own dataset much you would do in a relational database to manage data volumes.

# %%

def final_market_book(s):
    with patch("builtins.open", lambda f, _: f):
        gen = s.get_generator()
        for market_books in gen():
            # Check if this market book meets our market filter ++++++++++++++++++++++++++++++++++
            if ((evaluate_market := filter_market(market_books[0])) == False):
                    return(None)
            for market_book in market_books:
                last_market_book = market_book
        return(last_market_book)

def parse_final_selection_meta(dir, out_file):
    
    with open(out_file, "w+") as output:

        output.write("market_id,selection_id,venue,market_time,selection_name,win,bsp\n")

        for file_obj in load_markets(dir):

            stream = trading.streaming.create_historical_generator_stream(
                file_path=file_obj,
                listener=listener,
            )

            last_market_book = final_market_book(stream)
            if last_market_book is None:
                continue 

            # Extract Info ++++++++++++++++++++++++++++++++++
            runnerMeta = [
                {
                    'selection_id': r.selection_id,
                    'selection_name': next((rd.name for rd in last_market_book.market_definition.runners if rd.selection_id == r.selection_id), None),
                    'selection_status': r.status,
                    'win': np.where(r.status == "WINNER", 1, 0),
                    'sp': r.sp.actual_sp
                }
                for r in last_market_book.runners 
            ]

            # Return Info ++++++++++++++++++++++++++++++++++
            for runnerMeta in runnerMeta:
                if runnerMeta['selection_status'] != 'REMOVED':
                    output.write(
                        "{},{},{},{},{},{},{}\n".format(
                            str(last_market_book.market_id),
                            runnerMeta['selection_id'],
                            last_market_book.market_definition.venue,
                            last_market_book.market_definition.market_time,
                            runnerMeta['selection_name'],
                            runnerMeta['win'],
                            runnerMeta['sp']
                        )
                    )

# %%
selection_meta = "[OUTPUT PATH TO CSV FOR SELECTION METADATA]"
stream_files = glob.glob("[PATH TO STREAM FILES]*.tar")
# trading = betfairlightweight.APIClient("username", "password")
# listener = StreamListener(max_latency=None)

print("__ Parsing Selection Metadata ___ ")
# parse_final_selection_meta(stream_files, selection_meta)

# %% [markdown]
# ## 1.4 Preplay Prices and Projections
# 
# In this set of preplay prices I'm interested in many of the same fields as we've extracted in previous tutorials as well as fields relating to the current state of the BSP.
# 
# These objects sit under the `sp` slot within the returned `runner` object. The fields we'll extract are:
# 
# - The so called "near price"
#     + The near price is the projected SP value you can see on the website
#     + It includes both bets already placed into the SP pools as well as open limit orders to estimate what the final BSP value will be
# - The so called "far price"
#     + This is the same as the near price except it excludes limit orders on the exchange
#     + This makes it fairly redundant value and we'll see how poor of an estimator it is a bit later
# - The volume currently bet into the BSP back pool
# - The liability currently laid into the BSP lay pool
# 
# We'll also extract the top 5 rungs of the available to back and available to lay ladders as well as the traded volume of limit bets.
# 
# It's worth noting that I am discarding some key information about the BSP pools that I could have extracted if I wanted to. The current SP bets are laid out in a way that I could split out `limit_on_close` as well as `market_on_close` sp bets but I've rolled everything together in SP stake on the back side and sp liability on the lay side. This is just to reduce complexity of this article but including it would increase the predictive power of the BSP model in the final step.

# %%
def loop_preplay_prices(s, o):

    with patch("builtins.open", lambda f, _: f):

        gen = s.get_generator()

        marketID = None
        tradeVols = None
        time = None
        last_book_recorded = False
        prev_book = None

        for market_books in gen():

            # Check if this market book meets our market filter ++++++++++++++++++++++++++++++++++

            if ((evaluate_market := filter_market(market_books[0])) == False):
                    break
            
            for market_book in market_books:

                # Time Step Management ++++++++++++++++++++++++++++++++++

                if marketID is None:
                    # No market initialised
                    marketID = market_book.market_id
                    time =  market_book.publish_time
                elif market_book.inplay and last_book_recorded:
                    break
                else:
                                            
                    seconds_to_start = (market_book.market_definition.market_time - market_book.publish_time).total_seconds()

                    if seconds_to_start > 120:
                        # Too early before off to start logging prices
                        prev_book = market_book
                        continue
                    else:
                        
                        # Update data at different time steps depending on seconds to off
                        wait = 10

                        # New Market
                        if market_book.market_id != marketID:
                            last_book_recorded = False
                            marketID = market_book.market_id
                            time =  market_book.publish_time
                            continue
                        # (wait) seconds elapsed since last write
                        elif (market_book.publish_time - time).total_seconds() > wait:
                            time = market_book.publish_time
                        # if current marketbook is inplay want to record the previous market book as it's the last preplay marketbook
                        elif market_book.inplay:
                            last_book_recorded = True
                            market_book = prev_book
                        # fewer than (wait) seconds elapsed continue to next loop
                        else:
                            prev_book = market_book
                            continue

                # Execute Data Logging ++++++++++++++++++++++++++++++++++
                for runner in market_book.runners:

                    try:
                        atb_ladder = pull_ladder(runner.ex.available_to_back, n = 5)
                        atl_ladder = pull_ladder(runner.ex.available_to_lay, n = 5)
                    except:
                        atb_ladder = {}
                        atl_ladder = {}

                    limitTradedVol = sum([rung.size for rung in runner.ex.traded_volume])

                    o.writerow(
                        (
                            market_book.market_id,
                            runner.selection_id,
                            market_book.publish_time,
                            int(limitTradedVol),
                            # SP Fields
                            runner.sp.near_price,
                            runner.sp.far_price,
                            int(sum([ps.size for ps in runner.sp.back_stake_taken])),
                            int(sum([ps.size for ps in runner.sp.lay_liability_taken])),
                            # Limit bets available
                            str(atb_ladder).replace(' ',''), 
                            str(atl_ladder).replace(' ','')
                        )
                    )

                prev_book = market_book

def parse_preplay_prices(dir, out_file):
    
    with open(out_file, "w+") as output:

        writer = csv.writer(
            output, 
            delimiter=',',
            lineterminator='\r\n',
            quoting=csv.QUOTE_ALL
        )
        writer.writerow(("market_id","selection_id","time","traded_volume","near_price","far_price","bsp_back_pool_stake","bsp_lay_pool_liability","atb_ladder",'atl_ladder'))

        for file_obj in load_markets(dir):

            stream = trading.streaming.create_historical_generator_stream(
                file_path=file_obj,
                listener=listener,
            )

            loop_preplay_prices(stream, writer)

# %%
price = "[OUTPUT PATH TO CSV FOR SELECTION METADATA]"
stream_files = glob.glob("[PATH TO STREAM FILES]*.tar")
# trading = betfairlightweight.APIClient("username", "password")
# listener = StreamListener(max_latency=None)

print("__ Parsing Selection Prices ___ ")
# parse_final_selection_meta(stream_files, price)

# %% [markdown]
# # 2.0 Analysis
# 
# First step let's boot up the datasets we extracted in the previous steps and take a look at what we've managed to extract from the raw stream files.
# 
# ## 2.1 Load and Inspect
# 
# First we have the highlevel selection metadata as we have already seen in other tutorials

# %%
selection = pd.read_csv("[PATH TO YOUR SELECTION METADATA FILE]", dtype={'market_id': object, 'selection_id': object}, parse_dates = ['market_time'])

selection.head(3)

# %% [markdown]
# Now let's load the prices file. We'll apply some extra logic to parse the ladder columns into dictionaries and also remove the first odds record per group as it's the first record as the market was instantiated.

# %%
prices = pd.read_csv(
    "[PATH TO YOUR PRICES FILE]", 
    quoting=csv.QUOTE_ALL,
    dtype={'market_id': 'string', 'selection_id': 'string', 'atb_ladder': 'string', 'atl_ladder': 'string'},
    parse_dates=['time']
)

# Parse ladder columns
prices['atb_ladder'] = [ast.literal_eval(x) for x in prices['atb_ladder']]
prices['atl_ladder'] = [ast.literal_eval(x) for x in prices['atl_ladder']]

# Drop the first row within each group
prices = prices.drop(prices.groupby(['market_id', 'selection_id'],as_index=False).nth(0).index)

prices.head(3)

# %%
f'The shape of the prices data file is {prices.shape[0]} rows and {prices.shape[1]} columns'

# %%
# Let's have a look at the prices datafile for a distinct market and selection
prices.query('market_id == "1.183995724" and selection_id == "22832649"')

# %% [markdown]
# We can see some expected behaviour as we zoom in on a particular selection
# 
# - The traded volume increases on this selection as we get closer to the jump
# - The projected BSP (the `near_price` column) stays constant for a number of increments as its update is cached for 60 seconds at a time
# - The sizes in the BSP pools also increases as we get closer to the jump
# - The prices offered and traded closer to the jump are closer to the BSP than those at the start of the 2 minute period

# %% [markdown]
# ## 2.2 Transform and Assemble
# 
# We have our 2 core datasets, but we'd prefer to work with one now. We'd also like to add some key columns that will be reused throughout our analysis so we'll add those now too.
# 
# 
# ### 2.2.1 Mid points
# 
# The first semi-interesting thing we'll do in this analysis is add selection mid-points to our dataset. Eventually we're going to be interested in estimating the BSP and measuring the efficiency of certain prices at various points leading up to the race. 
# 
# Betfair markets work like all markets with bids and spreads. The market equilibrium forms around the best price offered on either side of the market to back or to lay. These top prices each have some inherent advantage built into it for the offerer. For example in early markets the best offers on either side of the market might be really wide (say 1.80 as a best back and 2.50 as a best lay). Given the price discovery process is still immature each bidder gets a large premium, backed into their offer price, which compensates them for providing betting opportunities with little to no information provided from other market participants. This spread will naturally get tighter and tighter as the market matures and more participants seek to get volume down and must be more and more competitive. But what's the price "equilibrium" in each case?
# 
# Well it's up to you but I'll provide you two ways of finding the central mid-point of a bid-ask spread on betfair markets. The problem we're solving for here is the non-linearity of prices in odds space. We have some intuition for this: when we see a market spread of 10-100 in early trading we have an understanding that the true midpoint of this market is somewhere around 25-35 not the 55 you'd get if you simply took the (arithmetic) mean of those two numbers.
# 
# Two techniquest for accounting for that non-linearity are as follows.
# 
# **Ladder Midpoint**
# 
# The ladder midpoint method takes advantage of the fact that the betfair price ladder itself accounts for the nonlinearity of prices in odds space. The method calculated the difference in number of rungs on the betfair ladder, halves it, and shifts the best back or lay price that number of rungs towards the centre. This will generally provide a much better idea of the market midpoint than a simple arithmetic mean of the two prices.
# 
# **Geometric Mean**
# 
# Unfortunately the ladder method is a little computationally expensive. A good approximation for this approach is to take the geometric mean of the best back and best lay values. The geometric mean is a special kind of mean that you may have never used before that is more appropriate for purposes like this. It is calculated like: `sqrt(x1 * x2 * ...)`. This number will also provide a much better estimate of the market midpoint than the simple arithmetic mean.
# 
# The latter calculation is trivial. The former requires a suite of betfair tick arithmetic functions that I'll put below. It may seem like overkill for this exercise (and it is) but hopefully these functions might be of use to you for other purposes.

# %%
# Define the betfair tick ladder
def bfTickLadder():

    tickIncrements = {
        1.0: 0.01,
        2.0: 0.02,
        3.0: 0.05,
        4.0: 0.1,
        6.0: 0.2,
        10.0: 0.5,
        20.0: 1.0,
        30.0: 2.0,
        50.0: 5.0,
        100.0: 10.0,
        1000.0: 1000,
    }

    ladder = []

    for index, key in enumerate(tickIncrements):

        increment = tickIncrements[key]

        if (index+1) == len(tickIncrements):
            ladder.append(key)
        else:
            key1 = [*tickIncrements][index]
            key2 = [*tickIncrements][index+1]
            steps = (key2 - key1) / increment

            for i in range(int(steps)):
                ladder.append(round(key + i * increment, 2))

    return(ladder)

bfticks = bfTickLadder()
    
# Round a decimal to the betfair tick value below
def bfTickFloor(price, includeIndex=False):

    if 'bfticks' in globals():
        global bfticks
    else:
        bfticks = bfTickLadder()

    ind = [ n for n,i in enumerate(bfticks) if i>=price][0]
    if includeIndex:
        if bfticks[ind]==price:
            return((ind, price))
        else:
            return((ind-1, bfticks[ind-1]))
    else:
        if bfticks[ind]==price:
            return(price)
        else:
            return(bfticks[ind-1])

# Calculate the numder of ticks between two tick values
def bfTickDelta(p1, p2):

    if np.isnan(p1) or np.isnan(p2):
        return(np.nan)

    x = bfTickFloor(p1, includeIndex=True)
    y = bfTickFloor(p2, includeIndex=True)
    return(x[0]-y[0])

def bfTickShift(p, rungs):

    if 'bfticks' in globals():
        global bfticks
    else:
        bfticks = bfTickLadder()
    
    flr = bfTickFloor(p, includeIndex = True)

    return(bfticks[flr[0]+rungs])


def bfLadderMidPoint(p1, p2):

    if np.isnan(p1) or np.isnan(p2):
        return(np.nan)

    delta = -1 * bfTickDelta(p1, p2)

    if delta == 1:
        return(p1)
    elif delta % 2 != 0:
        return(bfTickShift(p1, math.ceil(delta / 2)))
    else:
        return(bfTickShift(p1, math.floor(delta / 2)))

# %%
# Let's test a midpoint using the ladder mid point method
bfLadderMidPoint(10,100)

# %%
# And for illustrative purposes let's calculate the geomtric mean of these values
np.sqrt(10 * 100)

# %% [markdown]
# Let's put this all together while stitching together our two core datasets.

# %%
# Join and augment
df = (
    selection.merge(prices, on = ['market_id', 'selection_id'])
    .assign(sbsj = lambda x: round((x['market_time'] - x['time']).dt.total_seconds() / 10) * 10)
    .assign(back_best = lambda x: [np.nan if d.get('p') is None else d.get('p')[0] for d in x['atb_ladder']])
    .assign(lay_best = lambda x: [np.nan if d.get('p') is None else d.get('p')[0] for d in x['atl_ladder']])
    .assign(geometric_mid_point = lambda x: round(1 / np.sqrt((1/x['back_best']) * (1/x['lay_best'])), 3))
    .assign(ladder_mid_point = lambda x: x.apply(lambda x: bfLadderMidPoint(x.back_best, x.lay_best), axis=1))
    .replace([np.inf, -np.inf], np.nan)
)

df.head(3)

# %% [markdown]
# ## 2.3 Analysing The BSP
# 
# Before we embark on our predictive exercise let's analyse the BSP to get a feel for it as an entity.
# 
# ### 2.3.1 Volumes
# 
# Ever wondered how much volume is traded on the BSP? How does it compare to limit bets? Well with our parsed stream data we can answer those questions! Now the BSP volume will be the bigger of the BSP back stake and the lay stake (which you can infer by the final BSP and the total lay liability).
# 
# 

# %%
# Volume Traded
# _________________________


# Extract the final time slice of data which includes the total preplay volumes traded across limit and BSP poools
volumeDf = df.groupby(['market_id', 'selection_id'],as_index=False).nth(-1)[['market_id', 'selection_id', 'bsp',  'traded_volume', 'bsp_back_pool_stake', 'bsp_lay_pool_liability']]

# Infer the biggest of the two BSP stakes
volumeDf = (
    volumeDf
    .assign(lay_stake = lambda x: x['bsp_lay_pool_liability'] / (x['bsp']-1))
    .assign(bsp_stake = lambda x: x[['lay_stake', 'bsp_back_pool_stake']].max(axis = 1))
)

(
    volumeDf
    .groupby('market_id', as_index = False)
    .agg({'traded_volume': 'sum', 'bsp_stake': 'sum'})
    .agg({'traded_volume': 'mean', 'bsp_stake': 'mean'})
)

# %% [markdown]
# So in an average thoroughbred market there's about 98k traded limit volume and 7,300 BSP traded stake. So approximately 7% of thoroughbred volume is traded at the BSP at least for our sample of thoroughbred races.
# 
# 
# ## 2.3.2 Efficiency?
# 
# Now you may have heard this story before: **you can't beat the BSP it's too efficient!**. I'm not sure people really have a firm idea about what they're talking about when they say this.
# 
# Typically what you'll see in a discussion about efficiency is the predicted vs observed scatterplot. Let's see if we can reproduce this chart.
# 
# First let's assemble a dataframe that we can use for this chart as well as others. What we'll do is we'll extract the BSP and a price value at 5 different slices before the race starts. We could chose any price point (we'll analyse the difference between them in a subsequent step) but for this section I'm going to take the preplay market estimate as the geometric market midpoint (you'll have to trust me for now that this is a sensible decision).

# %%
# Extract the geomtric market mid point at time slices: 120, 90, 60, 30, and 0 seconds from the scheduled off
preplay = df[df.sbsj.isin([120,90,60,30,0])][['market_id', 'selection_id', 'win', 'sbsj', 'geometric_mid_point']].sort_values(['market_id', 'selection_id', 'sbsj'], ascending = [True, True, False]).rename(columns={'geometric_mid_point': 'odds'}).assign(type = lambda x: "seconds before off: " + x['sbsj'].astype(int).astype(str))

# Extract the BSP values
bsp = df.sort_values(['market_id', 'selection_id', 'time'], ascending = [True, True, False]).groupby(['market_id', 'selection_id']).head(1)[['market_id', 'selection_id', 'win', 'sbsj', 'bsp']].rename(columns={'bsp': 'odds'}).assign(type = "bsp")

# Append them together
accuracyFrame = pd.concat([preplay, bsp]).dropna()
accuracyFrame.head(5)

# %% [markdown]
# Now we'll filter just on our BSP records and plot the observed vs actual scatterplot

# %%
# BSP Scatter
# __________________

winRates = (
    accuracyFrame
    .query('type == "bsp"')
    .assign(implied_chance = lambda x: round(20 * (1 / x['odds']))/20)
    .groupby('implied_chance', as_index = False)
    .agg({'win': 'mean'})
)

fig = px.scatter(winRates, x = "implied_chance", y = "win", template = "plotly_white", title = "BSP: implied win vs actual win")
fig.add_trace(
    go.Scatter(
        x = winRates.implied_chance, y = winRates.implied_chance, name = 'no bias', line_color = 'rgba(8,61,119, 0.3)'
    )
)
fig.show("png")

# %% [markdown]
# Ok aside from some small sample noise at the top end (there's very few horses that run at sub 1.20 BSPs) we can see that the BSP is pretty perfectly.... efficient? Is that the right word? I'd argue that it's very much not the right word. Let me illustrate with a counter example. Let's plot the same chart for the BSP as well as our 5 other price points.

# %%
# Bsp + Other Odds Scatter
# __________________

winRates = (
    accuracyFrame
    .assign(implied_chance = lambda x: round(20 * (1 / x['odds']))/20)
    .groupby(['type', 'implied_chance'], as_index = False)
    .agg({'win': 'mean'})
)

fig = px.scatter(winRates, x = "implied_chance", y = "win", color = 'type', template = "plotly_white", title = "Comparing Price Points: implied win vs actual win")
fig.add_trace(
    go.Scatter(
        x = winRates.implied_chance, y = winRates.implied_chance, name = 'no bias', line_color = 'rgba(8,61,119, 0.3)'
    )
)
fig.show("png")

# %% [markdown]
# So they're all efficient? And indecernibly as efficient as one another?
# 
# Well, to cut a long and possibly boring story short this isn't the right way to measure efficiency. What we're measure here is **bias**. All my scatter plot here tells me is if there's any systematic bias in the BSP, ie groups of BSPs that aren't well calibrated with actual outcomes. That is, for example, that perhaps randomly the group of horses that BSP around 2 don't happen to win 50% of the time maybe there was a sytemic bias that short favourites were underbet and these selections actually won 55% of the time. That would be a price bias in the BSP that someone could take advatange at just by looking at historical prices and outcomes alone.
# 
# For and even simpler counter point: I could create a perfectly well calibrated estimate that assigned a single odds value to every horse which was the overall horse empirical win rate over our sample: 10.25% (which is merely a reflection of field sizes). This estimate would be unbiased, and would pass through our scatterplot method unscathed but would it be an efficient estimate? Clearly not.

# %%
df.agg({'win': 'mean'})

# %% [markdown]
# Bias only tells us if there's a systematic way of exploiting the odds values themselves. I could have told you that this was unlikely but the scatterplot proves it.
# 
# How else could we measure efficiency? I propose using the `logloss` metric.
# 
# Let's calculate the logloss of the BSP

# %%
# Logloss ++++++++++++++++++++++++++++++++++++++++

from sklearn.metrics import log_loss

# Overall Logloss
# _________________

bspLoss = log_loss(
    y_pred = 1 / accuracyFrame.query('type == "bsp"')['odds'],
    y_true = accuracyFrame.query('type == "bsp"')['win']
)

print(f'The overall logloss of the BSP is {round(bspLoss,4)}')

# %% [markdown]
# Ok what does this mean? Well nothing really. This metric won't tell you anything by itself it's just useful for relative comparisons. Let's plot the logloss of our geometric midpoint at our various timeslices.
# 

# %%

# Logloss at Different Time Points
# _________________

accuracyFrame.groupby('type', as_index = False).apply(lambda x: log_loss(y_pred=1/x['odds'],y_true=x['win'])).rename(columns = {None: 'logloss'}).sort_values('logloss')

# %%
# And in chart form
fig = px.bar(
    accuracyFrame.groupby('type', as_index = False).apply(lambda x: log_loss(y_pred=1/x['odds'],y_true=x['win'])).rename(columns = {None: 'logloss'}).sort_values('logloss', ascending = False),
    x = "type",
    y = "logloss",
    template = "plotly_white",
    title = "Logloss Of Odds At Various Time Points"
)
fig.update_yaxes(range=[.2755, .2765])
fig.show("png")

# %% [markdown]
# Now this is a cool graph. This is exactly like we would have intiuited. The market sharpens monotonically as we approach the market jump with the BSP being the most effiecient of all the prices!
# 
# Hopefully you can now see the logical failing of measuring bias over market efficiency and it changes the way you think about your bet placement.
# 
# Let's move on to what we're here for: is it possible to predict the BSP.

# %% [markdown]
# ## 2.4 Predicting the BSP
# 
# Ok so I'm interested in finding the answer to the question: which estimate of BSP should i use when betting on the exchange and is it possible to beat the projected SP provided on the website and through the API?
# 
# Well the first thing we should recognise about this projection is that it's cached. What does that mean? It means it only updated every 60 seconds. This suprised me when i first learned it and it was actually causing issues in my bet placement logic for the SP.
# 
# Let's have a look at a selection to see how this works in practice

# %%
# Lets take a sample of a market and a selection
dSlice = df.query('market_id == "1.182394184" and selection_id == "39243409"').dropna()

# %%
def chartClosingPrices(d):

    fig = px.line(
        pd.melt(d[:-1][['sbsj', 'back_best', 'near_price']], id_vars = 'sbsj', var_name = 'price'), 
            x='sbsj', y='value',
            color = 'price',
            template='plotly_white',
            title="Selection",
            labels = {
                'sbsj': "Seconds Before Scheduled Jump"
            }
    )
    fig.update_layout(font_family="Roboto")
    fig.add_trace(
        go.Line(x = dSlice.sbsj, y = dSlice.bsp, name = 'BSP', line_color = 'rgba(8,61,119, 0.3)', mode = "lines")
    )
    fig['layout']['xaxis']['autorange'] = "reversed"
    fig.show("png")

chartClosingPrices(dSlice)

# %% [markdown]
# The red line is the projected BSP, you can see that it's not very responsive. As the best back price comes in from ~3 to 2.9 leading up to the jump the projected SP doesn't move because it's cached. If you were relying on this number for something important and you were using it in that period you were using stale information and you'd be worse off for it. In this instance the final SP was 2.79 so you may have made the wrong betting decision.
# 
# This is somewhat counter intuitive because the projected sp (the so called near price) should be a good estimate of the BSP because it synthetically runs the BSP algorithm on the current market state and produces and estimate, so you would think that it'd be a pretty good estimate.
# 
# Let's widen our sample a bit and see how it performs across our entire sample. We'll slice the data at the exact scheduled off and see how accurate various price points are at predicting what the final BSP is. We'll use mean absolute error (MAE) as our error metric. We'll assess 6 price points:
# 
# - The near price (projected sp)
# - The far price (projected sp excluding limit orders)
# - The best back price
# - The best lay price
# - The ladder midpoint price
# - The geometric midpoint price

# %%
# Measurement
# ________________________

estimatesDf = df[df.sbsj == 0][['bsp', 'near_price', 'far_price', 'back_best', 'lay_best', 'geometric_mid_point', 'ladder_mid_point']]

(
    pd.melt(estimatesDf, id_vars = 'bsp', var_name = 'estimate')
    .assign(error = lambda x: abs(x['value'] - x['bsp']) / x['bsp'])
    .groupby('estimate', as_index=False)
    .agg({'error': 'mean'})
    .sort_values('error')
)

# %% [markdown]
# So a bit surprisingly, in thoroughbred markets at the scheduled off your best to just use the current best back price as your estimate of the BSP. It significantly outperforms the projected SP and even some of our midpoint methods. 
# 
# Let's change the timeslice a little and take the very last moment before the market settles and see which performs best.

# %%
lastEstimatesDf = df.groupby(['market_id', 'selection_id'],as_index=False).nth(-1)[['bsp', 'near_price', 'far_price', 'back_best', 'lay_best', 'geometric_mid_point', 'ladder_mid_point']]

(
    pd.melt(lastEstimatesDf, id_vars = 'bsp', var_name = 'estimate')
    .assign(error = lambda x: abs(x['value'] - x['bsp']) / x['bsp'])
    .groupby('estimate', as_index=False)
    .agg({'error': 'mean'})
    .sort_values('error')
)


# %% [markdown]
# First thing to notice is the estimates get a lot better than at the scheduled off, as we'd expect. A bit surprisingly the projected SP is still very weak due to the caching issue. In this scenario the geometric mid point beforms significantly better than the current best back price which suggests that as the late market is forming the back and lay spread with start converging to the fair price and eventual BSP. I personally use the geometric midpoint as my BSP estimate as it's a quick and easy metric that performs pretty well.
# 
# What if you want more though? Is it possible to do better than these metrics? These simple price points use no information about what's in the BSP pools, surely if we used this information we'd be able to do better. Let's try to use machine learning to synthesise all this information at once.
# 
# ### 2.3.4 Machine Learning
# 
# We'll build a quite random forest model to estimate the BSP with current price and pool size information. This is a very simple application of machine learning so hopefully gives you an idea of its power without being too complex.
# 
# Now we need an intelligent way of turning our pool and ladder information into a feature to insert into our model, how could we engineer this feature? Well what we'll do is calculate a WAP required to fill our pool stake on the back and lay side. What does that mean? Say we've got $200 sitting in the BSP back pool and $200 sitting on the top box of 2.5 on the back side, in this instance our WAP value would be exactly 2.5 cause we can fill it all at the top box. But if however, there was only $100 in the top box then we'd need to move down the ladder to fill the remaining $100 volume. Our feature will simulate this allocation logic and return the final weighted average price required to fill the total BSP pool. Here's the functions to do it on the back and lay side respectively:

# %%
def wapToGetBack(pool, ladder):
    price = ladder['p']
    volume = ladder['v']
    try:
        indmax = min([ i for (i,j) in enumerate(cVolume) if j > pool ])+1
    except:
        indmax = len(volume)
    return(round(sum([a * b for a, b in zip(price[:indmax], volume[:indmax])]) / sum(volume[:indmax]),4))

def wapToGetLay(liability_pool, ladder):
    price = ladder['p']
    volume = ladder['v']
    liability = [(a-1) * b for a, b in zip(price, volume)]
    cLiability = np.cumsum(liability)
    try:
        indmax = min([ i for (i,j) in enumerate(cLiability) if j > liability_pool ])+1
    except:
        indmax = len(volume)
    return(round(sum([a * b for a, b in zip(price[:indmax], volume[:indmax])]) / sum(volume[:indmax]),4))


# %% [markdown]
# Now we'll set up our model matrix which will be the market state at the exact scheduled off. We'll also add our custom features.

# %%
model_matrix = df[['sbsj', 'atb_ladder', 'atl_ladder','bsp', 'traded_volume', 'near_price', 'far_price', 'bsp_back_pool_stake', 'bsp_lay_pool_liability', 'back_best', 'lay_best', 'geometric_mid_point', 'ladder_mid_point']]

# Filter at scheduled jump
model_matrix = model_matrix[model_matrix.sbsj == 0].dropna()

model_matrix = (
    model_matrix
    .assign(wap_to_get_back_pool = lambda x: x.apply(lambda x: wapToGetBack(x.bsp_back_pool_stake, x.atb_ladder), axis=1))
    .assign(wap_to_get_lay_pool = lambda x: x.apply(lambda x: wapToGetLay(x.bsp_lay_pool_liability, x.atl_ladder), axis=1))
)

# Drop other columns
model_matrix.drop(columns = ['sbsj', 'atb_ladder', 'atl_ladder'], inplace = True)

model_matrix.head(3)

# %% [markdown]
# Now the machine learning. Sklearn make this very simple, in our case it's a few lines only. We'll split our data into train and test sets and train a small random forrest to predict the BSP.

# %%
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# Setup Train / Test
train_features, test_features, train_labels, test_labels = train_test_split(model_matrix.drop(columns = ['bsp']), model_matrix['bsp'], test_size = 0.25)

print('Training Features Shape:', train_features.shape)
print('Training Labels Shape:', train_labels.shape)
print('Testing Features Shape:', test_features.shape)
print('Testing Labels Shape:', test_labels.shape)

# Instantiate Model
rf = RandomForestRegressor(n_estimators = 100)

# Train Model
rf.fit(train_features, train_labels)

# %% [markdown]
# Let's check out our predictions on the test set (remember our model hasn't seen any of this data so it should be a true reflection on how we'd perform on some new races that would happen this afternoon say)

# %%
# Use the forest's predict method on the test data
predicted_bsp = rf.predict(test_features)
predicted_bsp

# %% [markdown]
# Seems reasonable. All well and good though is the prediction any good? Let's measure it using MAE in the same way as we did before.

# %%
# Let's test our estimate vs our others in the same way as before

testDf = test_features
testDf['bsp'] = test_labels
testDf['rf_bsp_prediction'] = predicted_bsp


(
    pd.melt(testDf[['bsp', 'near_price', 'far_price', 'back_best', 'lay_best', 'geometric_mid_point', 'ladder_mid_point', 'rf_bsp_prediction']], id_vars = 'bsp', var_name = 'estimate')
    .assign(error = lambda x: abs(x['value'] - x['bsp']) / x['bsp'])
    .groupby('estimate', as_index=False)
    .agg({'error': 'mean'})
    .sort_values('error')
)

# %% [markdown]
# Nice that's significantly better than the best previous estimate at this time slice. To validate it further let's use the same model to predict the BSP using the market state 10 seconds after the scheduled jump instead of at the exact scheduled off. None of the rows (or samples) in this time slice have been seen by the model during the training step so it should provide a robust out of sample estimate of the models performance on unseen data.

# %%
# Validate it on a completely different time point - 10 seconds after scheduled jump

outOfSample = df[['sbsj', 'atb_ladder', 'atl_ladder','bsp', 'traded_volume', 'near_price', 'far_price', 'bsp_back_pool_stake', 'bsp_lay_pool_liability', 'back_best', 'lay_best', 'geometric_mid_point', 'ladder_mid_point']]

outOfSample = outOfSample[outOfSample.sbsj == -10].dropna()

outOfSample = (
    outOfSample
    .assign(wap_to_get_back_pool = lambda x: x.apply(lambda x: wapToGetBack(x.bsp_back_pool_stake, x.atb_ladder), axis=1))
    .assign(wap_to_get_lay_pool = lambda x: x.apply(lambda x: wapToGetLay(x.bsp_lay_pool_liability, x.atl_ladder), axis=1))
)

# Produce Predictions
outofsamplebspprediction = rf.predict(outOfSample.drop(columns = ['bsp', 'sbsj', 'atb_ladder', 'atl_ladder']))
outofsamplebspprediction

# %%
outOfSample['rf_bsp_prediction'] = outofsamplebspprediction

(
    pd.melt(outOfSample[['bsp', 'near_price', 'far_price', 'back_best', 'lay_best', 'geometric_mid_point', 'ladder_mid_point', 'rf_bsp_prediction']], id_vars = 'bsp', var_name = 'estimate')
    .assign(error = lambda x: abs(x['value'] - x['bsp']) / x['bsp'])
    .groupby('estimate', as_index=False)
    .agg({'error': 'mean'})
    .sort_values('error')
)

# %% [markdown]
# Still significantly better on the out of sample set which is a really positive sign.
# 
# ## 2.3.5 Next Steps
# 
# To improve this model I'd include multiple time slices in the training sample and use the seconds before scheduled jump as a feature as I would estimate that the predictive dynamics of each of these features is dynamic and affected by how mature + how close to settlement the market is. 
# 
# To implement this model in your bet placement code you'd simply need to save the model object (some info about how to do this with sklearn can be found here [here](https://scikit-learn.org/stable/modules/model_persistence.html)). Your key challenge will be making sure you can produce the exact inputs you've created in this development process from the live stream or polling API responses, but if you've gotten this far it won't be a huge challenge for you.
# 
# # 3.0 Conclusion
# 
# I've taken you through a quick crash course in the Betfair BSP including:
# 
# - What it is
# - How it's created
# - How it's traded on betfair Australian thoroughbred markets
# - How efficient it is and a methodology for measuring its efficiency in different contexts
# - The accuracy of the projected SP and how it compares with other estimates
# - How to build your own custom projection that's better than anything available out of the box
# 
# The analysis focused on thoroughbred markets but could easily be extended to other racing codes or sports markets that have BSP enabled. The custom SP projection methodology could be used for anything from staking your model more accurately or with some improvement maybe as part of a automated trading strategy.


