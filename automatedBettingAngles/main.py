import requests
import pandas as pd
from datetime import date, timedelta
import numpy as np
import os
import re
import tarfile
import zipfile
import bz2
import glob
import logging
import yaml
from unittest.mock import patch
from typing import List, Set, Dict, Tuple, Optional
from itertools import zip_longest
import betfairlightweight
from betfairlightweight import StreamListener
from betfairlightweight.resources.bettingresources import (
    PriceSize,
    MarketBook
)
from scipy.stats import t
import plotly.express as px


# Utility Functions
#   + Stream Parsing
#   + Betfair Race Data Scraping
#   + Various utilities
# _________________________________

def as_str(v) -> str:
    return '%.2f' % v if type(v) is float else v if type(v) is str else ''

def split_anz_horse_market_name(market_name: str) -> (str, str, str):
    parts = market_name.split(' ')
    race_no = parts[0] # return example R6
    race_len = parts[1] # return example 1400m
    race_type = parts[2].lower() # return example grp1, trot, pace
    return (race_no, race_len, race_type)

def filter_market(market: MarketBook) -> bool: 
    d = market.market_definition
    return (d.country_code == 'AU' 
        and d.market_type == 'WIN' 
        and (c := split_anz_horse_market_name(d.name)[2]) != 'trot' and c != 'pace')

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

def slicePrice(l, n):
    try:
        x = l[n].price
    except:
        x = np.nan
    return(x)

def sliceSize(l, n):
    try:
        x = l[n].size
    except:
        x = np.nan
    return(x)

def wapPrice(l, n):
    try:
        x = round(sum( [rung.price * rung.size for rung in l[0:(n-1)] ] ) / sum( [rung.size for rung in l[0:(n-1)] ]),2)
    except:
        x = np.nan
    return(x)

def ladder_traded_volume(ladder):
    return(sum([rung.size for rung in ladder]))

# Core Execution Fucntions
# _________________________________

def extract_components_from_stream(s):
    
    with patch("builtins.open", lambda f, _: f):   
    
        evaluate_market = None
        prev_market = None
        postplay = None
        preplay = None
        t5m = None
        t30s = None
        inplay_min_lay = None

        gen = s.get_generator()

        for market_books in gen():
            
            for market_book in market_books:

                # If markets don't meet filter return None's
                if evaluate_market is None and ((evaluate_market := filter_market(market_book)) == False):
                    return (None, None, None, None, None, None)

                # final market view before market goes in play
                if prev_market is not None and prev_market.inplay != market_book.inplay:
                    preplay = market_book

                # final market view before market goes is closed for settlement
                if prev_market is not None and prev_market.status == "OPEN" and market_book.status != prev_market.status:
                    postplay = market_book

                # Calculate Seconds Till Scheduled Market Start Time
                seconds_to_start = (market_book.market_definition.market_time - market_book.publish_time).total_seconds()
                    
                # Market at 30 seconds before scheduled off
                if t30s is None and seconds_to_start < 30:
                    t30s = market_book

                # Market at 5 mins before scheduled off
                if t5m is None and seconds_to_start < 5*60:
                    t5m = market_book

                # Manage Inplay Vectors
                if market_book.inplay:

                    if inplay_min_lay is None:
                        inplay_min_lay = [ slicePrice(runner.ex.available_to_lay,0) for runner in market_book.runners]
                    else:
                        inplay_min_lay = np.fmin(inplay_min_lay, [ slicePrice(runner.ex.available_to_lay,0) for runner in market_book.runners])

                # update reference to previous market
                prev_market = market_book

        # If market didn't go inplay
        if postplay is not None and preplay is None:
            preplay = postplay
            inplay_min_lay = ["" for runner in market_book.runners]

        return (t5m, t30s, preplay, postplay, inplay_min_lay, prev_market) # Final market is last prev_market

def parse_stream(stream_files, output_file):
    
    with open(output_file, "w+") as output:

        output.write("market_id,selection_id,selection_name,wap_5m,wap_30s,bsp,ltp,traded_vol,inplay_min_lay\n")

        for file_obj in load_markets(stream_files):

            stream = trading.streaming.create_historical_generator_stream(
                file_path=file_obj,
                listener=listener,
            )

            (t5m, t30s, preplay, postplay, inplayMin, final) = extract_components_from_stream(stream)

            # If no price data for market don't write to file
            if postplay is None or final is None or t30s is None:
                continue; 

            # All runner removed
            if all(runner.status == "REMOVED" for runner in final.runners):
                continue

            runnerMeta = [
                {
                    'selection_id': r.selection_id,
                    'selection_name': next((rd.name for rd in final.market_definition.runners if rd.selection_id == r.selection_id), None),
                    'selection_status': r.status,
                    'sp': r.sp.actual_sp
                }
                for r in final.runners 
            ]

            ltp = [runner.last_price_traded for runner in preplay.runners]

            tradedVol = [ ladder_traded_volume(runner.ex.traded_volume) for runner in postplay.runners ]

            wapBack30s = [ wapPrice(runner.ex.available_to_back, 3) for runner in t30s.runners]

            wapBack5m = [ wapPrice(runner.ex.available_to_back, 3) for runner in t5m.runners]

            # Writing To CSV
            # ______________________

            for (runnerMeta, ltp, tradedVol, inplayMin, wapBack5m, wapBack30s) in zip(runnerMeta, ltp, tradedVol, inplayMin, wapBack5m, wapBack30s):
                
                if runnerMeta['selection_status'] != 'REMOVED':

                    output.write(
                        "{},{},{},{},{},{},{},{},{}\n".format(
                            str(final.market_id),
                            runnerMeta['selection_id'],
                            runnerMeta['selection_name'],
                            wapBack5m,
                            wapBack30s,
                            runnerMeta['sp'],
                            ltp,
                            round(tradedVol),
                            inplayMin
                        )
                    )

def get_bf_markets(dte):

    url = 'https://apigateway.betfair.com.au/hub/racecard?date={}'.format(dte)

    responseJson = requests.get(url).json()

    marketList = []

    for meeting in responseJson['MEETINGS']:
        for markets in meeting['MARKETS']:
            marketList.append(
                {
                    'date': dte,
                    'track': meeting['VENUE_NAME'],
                    'country': meeting['COUNTRY'],
                    'race_type': meeting['RACE_TYPE'],
                    'race_number': markets['RACE_NO'],
                    'market_id': str('1.' + markets['MARKET_ID']),
                    'start_time': markets['START_TIME']
                }
            )
    
    marketDf = pd.DataFrame(marketList)

    return(marketDf)

def get_bf_race_meta(market_id):

    url = 'https://apigateway.betfair.com.au/hub/raceevent/{}'.format(market_id)

    responseJson = requests.get(url).json()

    if 'error' in responseJson:
        return(pd.DataFrame())

    raceList = []

    for runner in responseJson['runners']:

        if 'isScratched' in runner and runner['isScratched']:
            continue

        # Jockey not always populated
        try:
            jockey = runner['jockeyName']
        except:
            jockey = ""

        # Place not always populated
        try:
            placeResult = runner['placedResult']
        except:
            placeResult = ""

        # Place not always populated
        try:
            trainer = runner['trainerName']
        except:
            trainer = ""

        raceList.append(
            {
                'market_id': market_id,
                'weather': responseJson['weather'],
                'track_condition': responseJson['trackCondition'],
                'race_distance': responseJson['raceLength'],
                'selection_id': runner['selectionId'],
                'selection_name': runner['runnerName'],
                'barrier': runner['barrierNo'],
                'place': placeResult,
                'trainer': trainer,
                'jockey': jockey,
                'weight': runner['weight']
            }
        )

    raceDf = pd.DataFrame(raceList)

    return(raceDf)

def scrape_thoroughbred_bf_date(dte):

    markets = get_bf_markets(dte)

    if markets.shape[0] == 0:
        return(pd.DataFrame())

    thoMarkets = markets.query('country == "AUS" and race_type == "R"')

    if thoMarkets.shape[0] == 0:
        return(pd.DataFrame())

    raceMetaList = []

    for market in thoMarkets.market_id:
        raceMetaList.append(get_bf_race_meta(market))

    raceMeta = pd.concat(raceMetaList)

    return(markets.merge(raceMeta, on = 'market_id'))


# Execute Data Pipeline
# _________________________________

# Description:
#   Will loop through a set of dates (starting July 2020 in this instance) and return race metadata from betfair 
# Estimated Time:
#   ~60 mins
# 
# if __name__ == '__main__':
    # dataList = []
    # dateList = pd.date_range(date(2020,7,1),date.today()-timedelta(days=1),freq='d')
    # for dte in dateList:
    #     dte = dte.date()
    #     print(dte)
    #     races = scrapeThoroughbredBfDate(dte)
    #     dataList.append(races)
    # data = pd.concat(dataList)
    # data.to_csv("[LOCAL PATH SOMEWHERE]", index=False)


# Description:
#   Will loop through a set of stream data archive files and extract a few key pricing measures for each selection
# Estimated Time:
#   ~6 hours
#
# trading = betfairlightweight.APIClient("username", "password")
# listener = StreamListener(max_latency=None)
# stream_files = glob.glob("[PATH TO LOCAL FOLDER STORING ARCHIVE FILES]*.tar")
# output_file = "[SOME OUTPUT DIRECTORY]/thoroughbred-odds-2021.csv"
# if __name__ == '__main__':
#     parse_stream(stream_files, output_file)


# Analysis
# _________________________________


# Functions ++++++++

def bet_eval_metrics(d, side = False):

    metrics = pd.DataFrame(d
    .agg({"npl": "sum", "stake": "sum", "win": "mean"})
    ).transpose().assign(pot=lambda x: x['npl'] / x['stake'])
    
    return(metrics[metrics['stake'] != 0])

def pl_pValue(number_bets, npl, stake, average_odds):

    pot = npl / stake

    tStatistic = (pot * np.sqrt(number_bets)) / np.sqrt( (1 + pot) * (average_odds - 1 - pot) )

    pValue = 2 * t.cdf(-abs(tStatistic), number_bets-1)

    return(np.where(np.logical_or(np.isnan(pValue), pValue == 0), 1, pValue))

def distance_group(distance):

    if distance is None:
        return("missing")
    elif distance < 1100:
        return("sprint")
    elif distance < 1400:
        return("mid_short")
    elif distance < 1800:
        return("mid_long")
    else:
        return("long")

def barrier_group(barrier):
    if barrier is None:
        return("missing")
    elif barrier < 4:
        return("inside")
    elif barrier < 9:
        return("mid_field")
    else:
        return("outside")

# Analysis ++++++++

# Local Paths (will be different on your machine)
path_odds_local = "[PATH TO YOUR LOCAL FILES]/thoroughbred-odds-2021.csv"
path_race_local = "[PATH TO YOUR LOCAL FILES]/thoroughbred-race-data.csv"

odds = pd.read_csv(path_odds_local, dtype={'market_id': object, 'selection_id': object})
race = pd.read_csv(path_race_local, dtype={'market_id': object, 'selection_id': object})

# Joining two datasets
df = race.merge(odds.loc[:, odds.columns != 'selection_name'], how = "inner", on = ['market_id', 'selection_id'])

# I'll also add columns for the net profit from backing and laying each selection to be picked up in subsequent sections
df['back_npl'] = np.where(df['place'] == 1, 0.95 * (df['bsp']-1), -1)
df['lay_npl'] = np.where(df['place'] == 1, -1 * (df['bsp']-1), 0.95)

# Adding Variable Chunks
df['distance_group'] = pd.to_numeric(df.race_distance, errors = "coerce").apply(distance_group)
df['barrier_group'] = pd.to_numeric(df.barrier, errors = "coerce").apply(barrier_group)

# Data Partitioning
dfTrain = df.query('date < "2021-04-01"')
dfTest = df.query('date >= "2021-04-01"')

'{} rows in the "training" set and {} rows in the "test" data'.format(dfTrain.shape[0], dfTest.shape[0])

# Angle 1 ++++++++++++++++++++++++++++++++++++++++++++++

(
    dfTrain
    .assign(stake=1)
    .groupby('selection_name', as_index = False)
    .agg({'back_npl': 'sum', 'stake': 'sum'})
    .assign(pot=lambda x: x['back_npl'] / x['stake'])
    .sort_values('pot', ascending=False)  
    .head(3) 
)

# Calculate the profit (back and lay) and average odds across all track / distance / barrier group combos
trackDistanceBarrier = (
    dfTrain
    .assign(stake = 1)
    .assign(odds = lambda x: x['bsp'])
    .groupby(['track', 'race_distance', 'barrier_group'], as_index=False)
    .agg({'back_npl': 'sum', 'lay_npl': 'sum','stake': 'sum', 'odds': 'mean'})
)

trackDistanceBarrier

trackDistanceBarrier = (
    trackDistanceBarrier
    .assign(backPL_pValue = lambda x: pl_pValue(number_bets = x['stake'], npl = x['back_npl'], stake = x['stake'], average_odds = x['odds']))
    .assign(layPL_pValue = lambda x: pl_pValue(number_bets = x['stake'], npl = x['lay_npl'], stake = x['stake'], average_odds = x['odds']))
)

trackDistanceBarrier

# Top 5 lay combos Track | Distance | Barrier (TDB)
TDB_bestLay = trackDistanceBarrier.query('lay_npl>0').sort_values('layPL_pValue').head(5)
TDB_bestLay

# First let's test laying on the train set (by definition we know these will be profitable)
train_TDB_bestLay = (
    dfTrain
    .merge(TDB_bestLay[['track', 'race_distance']])
    .assign(npl=lambda x: x['lay_npl'])
    .assign(stake=1)
    .assign(win=lambda x: np.where(x['lay_npl'] > 0, 1, 0))
)

# This is the key test (non of the races has been part of analysis to this point)
test_TDB_bestLay = (
    dfTest
    .merge(TDB_bestLay[['track', 'race_distance']])
    .assign(npl=lambda x: x['lay_npl'])
    .assign(stake=1)
    .assign(win=lambda x: np.where(x['lay_npl'] > 0, 1, 0))
)

# Peaking at the bets in the test set
test_TDB_bestLay[['track', 'race_distance', 'barrier', 'barrier_group', 'bsp', 'lay_npl', 'win', 'stake']]

# Let's run our evaluation on the training set
bet_eval_metrics(train_TDB_bestLay)

# And on the test set
bet_eval_metrics(test_TDB_bestLay)

# Angle 2 ++++++++++++++++++++++++++++++++++++++++++++++

(
    dfTrain
    .assign(market_support=lambda x: x['wap_5m'] / x['wap_30s'])
    .assign(races=1)
    .groupby('jockey')
    .agg({'market_support': 'mean', 'races': 'count'})
    .query('races > 10')
    .sort_values('market_support', ascending = False)
    .head()
)

# Group By Jockey and Market Support
jockeys = (
    dfTrain
    .assign(stake = 1)
    .assign(odds = lambda x: x['bsp'])
    .assign(npl=lambda x: np.where(x['place'] == 1, 0.95 * (x['odds']-1), -1))
    .assign(market_support=lambda x: np.where(x['wap_5m'] > x['wap_30s'], "Y", "N"))
    .groupby(['jockey', 'market_support'], as_index=False)
    .agg({'odds': 'mean', 'stake': 'sum', 'npl': 'sum'})
    .assign(pValue = lambda x: pl_pValue(number_bets = x['stake'], npl = x['npl'], stake = x['stake'], average_odds = x['odds']))
)

jockeys.sort_values('pValue').query('npl > 0').head(10)

# First evaluate on our training set
train_jockeyMarket = (
    dfTrain
    .assign(market_support=lambda x: np.where(x['wap_5m'] > x['wap_30s'], "Y", "N"))
    .merge(jockeys.sort_values('pValue').query('npl > 0').head(10)[['jockey', 'market_support']])
    .assign(stake = 1)
    .assign(odds = lambda x: x['bsp'])
    .assign(npl=lambda x: np.where(x['place'] == 1, 0.95 * (x['odds']-1), -1))
    .assign(win=lambda x: np.where(x['npl'] > 0, 1, 0))
)

# And on the test set
test_jockeyMarket = (
    dfTest
    .assign(market_support=lambda x: np.where(x['wap_5m'] > x['wap_30s'], "Y", "N"))
    .merge(jockeys.sort_values('pValue').query('npl > 0').head(10)[['jockey', 'market_support']])
    .assign(stake = 1)
    .assign(odds = lambda x: x['bsp'])
    .assign(npl=lambda x: np.where(x['place'] == 1, 0.95 * (x['odds']-1), -1))
    .assign(win=lambda x: np.where(x['npl'] > 0, 1, 0))
)

bet_eval_metrics(train_jockeyMarket)

bet_eval_metrics(test_jockeyMarket)

# Angle 3 ++++++++++++++++++++++++++++++++++++++++++++++


# First Investigate The Average Inplay Minimums And Loss Rates of Certain Jockeys
tradeOutIndex = (
    dfTrain
    .query('distance_group in ["long", "mid_long"]')
    .assign(inplay_odds_ratio=lambda x: x['inplay_min_lay'] / x['bsp'])
    .assign(win=lambda x: np.where(x['place']==1,1,0))
    .assign(races=lambda x: 1)
    .groupby(['jockey'], as_index=False)
    .agg({'inplay_odds_ratio': 'mean', 'win': 'mean', 'races': 'sum'})
    .sort_values('inplay_odds_ratio')
    .query('races >= 5')
)

tradeOutIndex

targetTradeoutFraction = 0.5

train_JockeyBackToLay = (
    dfTrain
    .query('distance_group in ["long", "mid_long"]')
    .merge(tradeOutIndex.head(20)['jockey'])
    .assign(npl=lambda x: np.where(x['inplay_min_lay'] <= targetTradeoutFraction * x['bsp'], np.where(x['place'] == 1, 0.95 * (x['bsp']-1-(0.5*x['bsp']-1)), 0), -1))
    .assign(stake=lambda x: np.where(x['npl'] != -1, 2, 1))
    .assign(win=lambda x: np.where(x['npl'] >= 0, 1, 0))
)

bet_eval_metrics(train_JockeyBackToLay)

test_JockeyBackToLay = (
    dfTest
    .query('distance_group in ["long", "mid_long"]')
    .merge(tradeOutIndex.head(20)['jockey'])
    .assign(npl=lambda x: np.where(x['inplay_min_lay'] <= targetTradeoutFraction * x['bsp'], np.where(x['place'] == 1, 0.95 * (x['bsp']-1-(0.5*x['bsp']-1)), 0), -1))
    .assign(stake=lambda x: np.where(x['npl'] != -1, 2, 1))
    .assign(win=lambda x: np.where(x['npl'] >= 0, 1, 0))

)

bet_eval_metrics(test_JockeyBackToLay)