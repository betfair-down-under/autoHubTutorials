import pandas as pd
import numpy as np
import os
import csv
import csv
import tarfile
import zipfile
import bz2
import glob
import ast
from unittest.mock import patch
import betfairlightweight
from betfairlightweight import StreamListener
from betfair_data import bflw
import pandas as pd
from betfair_data import PriceSize
import functools
from typing import List, Tuple
from pandas.errors import SettingWithCopyWarning
import warnings
from datetime import datetime
warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)
from itertools import zip_longest
from currency_converter import CurrencyConverter

file_directory= '' #INSERT FILE DIRECTORY WHERE TAR FILES ARE STORED

log1_Start = 60 * 10 # Seconds before scheduled off to start recording data for data segment one
log1_Step = 30       # Seconds between log steps for first data segment
log2_Start = 60 * 1  # Seconds before scheduled off to start recording data for data segment two
log2_Step = 10    # Seconds between log steps for second data segment

# splitting race name and returning the parts 
def split_anz_horse_market_name(market_name: str) -> Tuple[str, int, str]:
    # return race no, length, race type
    # input samples: 
    # 'R6 1400m Grp1' -> ('R6','1400m','grp1')
    # 'R1 1609m Trot M' -> ('R1', '1609m', 'trot')
    # 'R4 1660m Pace M' -> ('R4', '1660m', 'pace')
    parts = market_name.split(' ')
    race_no = parts[0] 
    race_len = parts[1].split('m')
    race_len = race_len[0]
    race_type = parts[2].lower() 
    return (race_no, race_len, race_type)

# filtering markets to those that fit the following criteria
def filter_market(market: bflw.MarketBook) -> bool: 
    d = market.market_definition
    return (d != None
        and d.country_code == 'AU' 
        and d.market_type == 'WIN'
        and (c := split_anz_horse_market_name(d.name)[2]) != 'trot' and c != 'pace' #strips out Harness Races
        )

# Simply add the below variable name to the market filter function above with the filter value
# Equals (== 'Value in Quotation' or True/False/None), Does Not Equal (!= 'Value in Quotation' or True/False/None) - FOR ALL TYPES
# Greater than (>), Greater than or equal to (>=), Less than (<), Less than or equal to (<=) - FOR INT/FLOAT
# For list of value 'in'

# and d.betting_type: str - ODDS, ASIAN_HANDICAP_SINGLES, ASIAN_HANDICAP_DOUBLES or LINE
# and d.bsp_market: bool - True, False
# and d.country_code: str - list of codes can be found here: https://en.wikipedia.org/wiki/ISO_3166-1_alpha-2 - Default value is 'GB' - Australia = 'AU', New Zealand = 'NZ'
# and d.event_id: str - PARENT_EVENT_ID
# and d.event_name: Optional[str] - Usually the name of the Match-Up (e.g. Bangladesh v Sri Lanka) or Race Meeting Name (e.g. Wangaratta (AUS) 1st Dec) - Note: Dictionaries don't support wildcard searches
# and d.event_type_id: str - SportID [Horse Racing - 7, Greyhounds - 4339]
# and d.market_base_rate: float - Market Commission Rate
# and d.market_type: str - e.g. "WIN"
# and d.name: Optional[str] - market name (e.g. R1 1170m Mdn)
# and d.number_of_active_runners: int - number of horses/dogs in the race
# and d.number_of_winners: int - Win market 1, Place markets 2+
# and d.turn_in_play_enabled: bool - True, False
# and d.venue: Optional[str] - Racing Only - Track

trading = betfairlightweight.APIClient(username = "username", password = "password", app_key="app_key")
listener = StreamListener(max_latency=None)

stream_files = glob.glob(file_directory+"*.tar") 
selection_meta = file_directory+"metadata.csv"
prices_path =  file_directory+"preplay.csv"

# rounding to 2 decimal places or returning '' if blank
def as_str(v) -> str:
    return '%.2f' % v if (type(v) is float) or (type(v) is int) else v if type(v) is str else ''

# returning smaller of two numbers where min not 0
def min_gr0(a: float, b: float) -> float:
    if a <= 0:
        return b
    if b <= 0:
        return a

    return min(a, b)

# parsing price data and pulling out weighted avg price, matched, min price and max price
def parse_traded(traded: List[PriceSize]) -> Tuple[float, float, float, float]:
    if len(traded) == 0: 
        return (None, None, None, None)

    (wavg_sum, matched, min_price, max_price) = functools.reduce(
        lambda total, ps: (
            total[0] + (ps.price * ps.size), # wavg_sum before we divide by total matched
            total[1] + ps.size, # total matched
            min(total[2], ps.price), # min price matched
            max(total[3], ps.price), # max price matched
        ),
        traded,
        (0, 0, 1001, 0) # starting default values
    )

    wavg_sum = (wavg_sum / matched) if matched > 0 else None # dividing sum of wavg by total matched
    matched = matched if matched > 0 else None 
    min_price = min_price if min_price != 1001 else None
    max_price = max_price if max_price != 0 else None

    return (wavg_sum, matched, min_price, max_price)


def load_markets(file_paths):
    for file_path in file_paths:
        print(file_path)
        print("__ Parsing Detailed Prices ___ ")
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
        x = ""
    return(x)

def sliceSize(l, n):
    try:
        x = l[n].size
    except:
        x = ""
    return(x)

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

def loop_prices(s, o):

    with patch("builtins.open", lambda f, _: f):

        gen = s.get_generator()

        marketID = None
        tradeVols = None
        time = None

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

                elif market_book.inplay:

                    # Stop once market inplay
                    break

                else:
                    
                    seconds_to_start = (market_book.market_definition.market_time - market_book.publish_time).total_seconds()

                    if seconds_to_start > log1_Start:
                        
                        # Too early before off to start logging prices
                        continue

                    else:
                    
                        # Update data at different time steps depending on seconds to off
                        wait = np.where(seconds_to_start <= log2_Start, log2_Step, log1_Step)

                        # New Market
                        if market_book.market_id != marketID:
                            marketID = market_book.market_id
                            time =  market_book.publish_time
                        # (wait) seconds elapsed since last write
                        elif (market_book.publish_time - time).total_seconds() > wait:
                            time = market_book.publish_time
                        # fewer than (wait) seconds elapsed continue to next loop
                        else:
                            continue

                # Execute Data Logging ++++++++++++++++++++++++++++++++++
                                                
                for runner in market_book.runners:

                    try:
                        selection_status = runner.status
                        reduction_factor = runner.adjustment_factor
                        atb_ladder = pull_ladder(runner.ex.available_to_back, n = 5)
                        atl_ladder = pull_ladder(runner.ex.available_to_lay, n = 5)
                        spn = runner.sp.near_price
                        spf = runner.sp.far_price
                    except:
                        selection_status = None
                        reduction_factor = None
                        atb_ladder = {}
                        atl_ladder = {}
                        spn = None
                        spf = None

                    # Calculate Current Traded Volume + Traded WAP
                    limitTradedVol = sum([rung.size for rung in runner.ex.traded_volume])
                    if limitTradedVol == 0:
                        limitWAP = ""
                    else:
                        limitWAP = sum([rung.size * rung.price for rung in runner.ex.traded_volume]) / limitTradedVol
                        limitWAP = round(limitWAP, 2)

                    o.writerow(
                        (
                            market_book.market_id,
                            market_book.number_of_active_runners,
                            runner.selection_id,
                            market_book.publish_time,
                            limitTradedVol,
                            limitWAP,
                            runner.last_price_traded or "",
                            selection_status,
                            reduction_factor,
                            str(atb_ladder).replace(' ',''), 
                            str(atl_ladder).replace(' ',''),
                            str(spn),
                            str(spf)
                        )
                    )


def parse_prices(dir, out_file):
    
    with open(out_file, "w+") as output:

        writer = csv.writer(
            output, 
            delimiter=',',
            lineterminator='\r\n',
            quoting=csv.QUOTE_ALL
        )
        
        writer.writerow(("market_id","active_runners","selection_id","time","traded_volume","wap","ltp","selection_status",'reduction_factor',"atb_ladder","atl_ladder","sp_near","sp_far"))

        for file_obj in load_markets(dir):

            stream = trading.streaming.create_historical_generator_stream(
                file_path=file_obj,
                listener=listener,
            )

            loop_prices(stream, writer)



#loop over each TAR file
for tar in stream_files:
    parse_prices([tar], prices_path)
    print("__ Parsing Market and Selection Data ___ ")

    # record prices to a file
    with open(selection_meta, "w") as output:
    # defining column headers
        output.write("market_id,event_date,country,track,event_name,selection_id,selection_name,result,bsp,pp_min,pp_max,pp_wap,pp_ltp,pp_volume,bsp_volume,ip_min,ip_max,ip_wap,ip_ltp,ip_volume\n")

        for i, g in enumerate(bflw.Files([tar])):
            print("Market {}".format(i), end='\r')

            def get_pre_post_final():
                eval_market = None
                prev_market = None
                preplay_market = None
                postplay_market = None       

                for market_books in g:
                    for market_book in market_books:
                        # if market doesn't meet filter return out
                        if eval_market is None and ((eval_market := filter_market(market_book)) == False):
                            return (None, None, None)

                        # final market view before market goes in play
                        if prev_market is not None and prev_market.inplay != market_book.inplay:
                            preplay_market = prev_market

                        # final market view at the conclusion of the market
                        if prev_market is not None and prev_market.status == "OPEN" and market_book.status != prev_market.status:
                            postplay_market = market_book

                        # update reference to previous market
                        prev_market = market_book

                return (preplay_market, postplay_market, prev_market) # prev is now final

            (preplay_market, postplay_market, final_market) = get_pre_post_final()

            # no price data for market
            if postplay_market is None:
                continue; 

            preplay_traded = [ (r.last_price_traded, r.ex.traded_volume) for r in preplay_market.runners ] if preplay_market is not None else None

            postplay_traded = [ (
                r.last_price_traded,
                r.ex.traded_volume,
                # calculating SP traded vol as smaller of back_stake_taken or (lay_liability_taken / (bsp - 1))        
                min_gr0(
                    next((pv.size for pv in r.sp.back_stake_taken if pv.size > 0), 0),
                    next((pv.size for pv in r.sp.lay_liability_taken if pv.size > 0), 0)  / ((r.sp.actual_sp if (type(r.sp.actual_sp) is float) or (type(r.sp.actual_sp) is int) else 0) - 1)
                ) if r.sp.actual_sp is not None else 0,
            ) for r in postplay_market.runners ]

            runner_data = [
            {
                'selection_id': r.selection_id,
                'selection_name': next((rd.name for rd in final_market.market_definition.runners if rd.selection_id == r.selection_id), None),
                'selection_status': r.status,
                'sp': as_str(r.sp.actual_sp),
            }
            for r in final_market.runners 
            ]

            # runner price data for markets that go in play
            if preplay_traded is not None:
                def runner_vals(r):
                    (pre_ltp, pre_traded), (post_ltp, post_traded, sp_traded) = r

                    inplay_only = list(filter(lambda ps: ps.size > 0, [
                        PriceSize(
                            price=post_ps.price, 
                            size=post_ps.size - next((pre_ps.size for pre_ps in pre_traded if pre_ps.price == post_ps.price), 0)
                        )
                        for post_ps in post_traded 
                    ]))

                    (ip_wavg, ip_matched, ip_min, ip_max) = parse_traded(inplay_only)
                    (pre_wavg, pre_matched, pre_min, pre_max) = parse_traded(pre_traded)

                    return {
                        'preplay_ltp': as_str(pre_ltp),
                        'preplay_min': as_str(pre_min),
                        'preplay_max': as_str(pre_max),
                        'preplay_wavg': as_str(pre_wavg),
                        'preplay_matched': as_str(pre_matched or 0),
                        'bsp_matched': as_str(sp_traded or 0),
                        'inplay_ltp': as_str(post_ltp),
                        'inplay_min': as_str(ip_min),
                        'inplay_max': as_str(ip_max),
                        'inplay_wavg': as_str(ip_wavg),
                        'inplay_matched': as_str(ip_matched),
                    }

                runner_traded = [ runner_vals(r) for r in zip_longest(preplay_traded, postplay_traded, fillvalue=PriceSize(0, 0)) ]

            # runner price data for markets that don't go in play
            else:
                def runner_vals(r):
                    (ltp, traded, sp_traded) = r
                    (wavg, matched, min_price, max_price) = parse_traded(traded)

                    return {
                        'preplay_ltp': as_str(ltp),
                        'preplay_min': as_str(min_price),
                        'preplay_max': as_str(max_price),
                        'preplay_wavg': as_str(wavg),
                        'preplay_matched': as_str(matched or 0),
                        'bsp_matched': as_str(sp_traded or 0),
                        'inplay_ltp': '',
                        'inplay_min': '',
                        'inplay_max': '',
                        'inplay_wavg': '',
                        'inplay_matched': '',
                    }

                runner_traded = [ runner_vals(r) for r in postplay_traded ]

            # printing to csv for each runner
            for (rdata, rprices) in zip(runner_data, runner_traded):
                # defining data to go in each column
                output.write(
                    "{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n".format(
                        postplay_market.market_id,
                        postplay_market.market_definition.market_time,
                        postplay_market.market_definition.country_code,
                        postplay_market.market_definition.venue,
                        postplay_market.market_definition.name,
                        rdata['selection_id'],
                        rdata['selection_name'],
                        rdata['selection_status'],
                        rdata['sp'],
                        rprices['preplay_min'],
                        rprices['preplay_max'],
                        rprices['preplay_wavg'],
                        rprices['preplay_ltp'],
                        rprices['preplay_matched'],
                        rprices['bsp_matched'],
                        rprices['inplay_min'],
                        rprices['inplay_max'],
                        rprices['inplay_wavg'],
                        rprices['inplay_ltp'],
                        rprices['inplay_matched'],
                    )
                )


    #loading selection file and parsing dates
    selection = pd.read_csv(selection_meta, dtype={'market_id': object, 'selection_id': object}, parse_dates = ['event_date'])

    #loading price file and parsing dates
    prices = pd.read_csv(
        prices_path, 
        quoting=csv.QUOTE_ALL,
        dtype={'market_id': 'string', 'selection_id': 'string', 'atb_ladder': 'string', 'atl_ladder': 'string'},
        parse_dates=['time']
    )

    #creating the ladder as a dictionary
    prices['atb_ladder'] = [ast.literal_eval(x) for x in prices['atb_ladder']]
    prices['atl_ladder'] = [ast.literal_eval(x) for x in prices['atl_ladder']]

    #merging the price and selection files
    df = selection.merge(prices, on = ['market_id', 'selection_id'])
    #assigning best prices available and calculating time relative to market start time
    df = (
        df
        .assign(back_best = lambda x: [np.nan if d.get('p') is None else d.get('p')[0] for d in x['atb_ladder']])
        .assign(lay_best = lambda x: [np.nan if d.get('p') is None else d.get('p')[0] for d in x['atl_ladder']])
        .assign(seconds_before_scheduled_off = lambda x: round((x['event_date'] - x['time']).dt.total_seconds()))
        .query('seconds_before_scheduled_off < @log1_Start')
    )

    #creating a unique list of market ids
    marketids = df['market_id'].unique().tolist()

    #writing each market to its own csv file
    for market in marketids:
        #create a dataframe and a naming convention for this market
        pricing_data=df[(df['market_id']==market)]
        if pricing_data.empty:
            continue
        race_track=pricing_data['track'].iloc[0]
        market_name=pricing_data['event_name'].iloc[0]
        market_time=pricing_data['event_date'].iloc[0]
        off=market_time.strftime('%Y-%m-%d')
        #write race details to the dataframe
        pricing_data['race']=pricing_data['event_name'].str.split('R').str[1]
        pricing_data['race']=pricing_data['race'].str.split(' ').str[0]
        pricing_data['distance']=pricing_data['event_name'].str.split(' ').str[1]
        pricing_data['distance']=pricing_data['distance'].str.split('m').str[0]
        pricing_data['race_type']=pricing_data['event_name'].str.split('m ').str[1]
        pricing_data['selection_name']=pricing_data['selection_name'].str.split('\. ').str[1]
        #convert GMT timezone to AEST/AEDT
        pricing_data['event_date']=pricing_data['event_date'].astype('datetime64[ns]')
        pricing_data['event_date']=pricing_data['event_date'].dt.tz_localize('UTC',ambiguous=False)
        pricing_data['event_date']=pricing_data['event_date'].dt.tz_convert('Australia/Melbourne')
        pricing_data['event_date']=pricing_data['event_date'].dt.tz_localize(None)
        pricing_data['time']=pricing_data['time'].astype('datetime64[ns]')
        pricing_data['time']=pricing_data['time'].dt.tz_localize('UTC',ambiguous=False)
        pricing_data['time']=pricing_data['time'].dt.tz_convert('Australia/Melbourne')
        pricing_data['time']=pricing_data['time'].dt.tz_localize(None)
        #covert GBP to AUD
        event_date=(pd.to_datetime(pricing_data['event_date']).dt.date).iloc[0]
        conversion_rate=CurrencyConverter(fallback_on_missing_rate=True).convert(1,'GBP','AUD',date=event_date)
        pricing_data['traded_volume']=pricing_data['traded_volume']*conversion_rate
        pricing_data.loc[(pricing_data['traded_volume'] < 0), 'traded_volume'] = 0
        pricing_data['traded_volume'] = pricing_data['traded_volume'].round(decimals=2)
        #reorder the dataframe and write to csv
        pricing_data=pricing_data[['event_date','country','track','race','distance','race_type','market_id','selection_id','selection_name',"selection_status",'reduction_factor','result','bsp','time','traded_volume','wap','ltp','atb_ladder','atl_ladder','back_best','lay_best','seconds_before_scheduled_off','sp_near','sp_far','pp_min','pp_max','pp_wap','pp_ltp','pp_volume','bsp_volume','ip_min','ip_max','ip_wap','ip_ltp','ip_volume']]
        pricing_data.to_csv(file_directory+off+' - '+race_track+' - '+market_name+'.csv',index=False)


#removing intermediate working documents to clean up
os.remove(selection_meta)
os.remove(prices_path)