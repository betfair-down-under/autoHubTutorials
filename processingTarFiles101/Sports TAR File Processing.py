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
from betfair_data import bflw #"Import "betfair_data.bflw" could not be resolved from source" - This is a known issue, the script should still run
import pandas as pd
from currency_converter import CurrencyConverter
from pandas.errors import SettingWithCopyWarning
import warnings
warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)

file_directory = ''# INSERT FILE DIRECTORY WHERE TAR FILES ARE STORED

log1_start = 60 * 60  # seconds before scheduled off to start recording data for data segment one
log1_step = 60  # seconds between log steps for first data segment
log2_start = 60 * 10  # seconds before scheduled off to start recording data for data segment two
log2_step = 10  # seconds between log steps for second data segment


def filter_market(market: bflw.MarketBook) -> bool:
    d = market.market_definition
    return (
        d is not None
        # and d.country_code in ['ES']
        # and d.market_type == 'MATCH_ODDS'
        and d.name in ['Match Odds', '1st Innings 20 Overs Line']
        # and d.betting_type == 'ODDS'
    )

# Simply add the below variable name to the market filter function above with the filter value
# Equals (== 'Value in Quotation' or True/False/None), Does Not Equal (!= 'Value in Quotation' or True/False/None) - FOR ALL TYPES
# Greater than (>), Greater than or equal to (>=), Less than (<), Less than or equal to (<=) - FOR INT/FLOAT
# For list of value 'in'

# and d.betting_type: str - ODDS, ASIAN_HANDICAP_SINGLES, ASIAN_HANDICAP_DOUBLES or LINE
# and d.bsp_market: bool - True, False
# and d.country_code: str - list of codes can be found here: https://en.wikipedia.org/wiki/ISO_3166-1_alpha-2 - Default value is 'GB' - Australia = 'AU', New Zealand = 'NZ'
# and d.event_id: str - PARENT_EVENT_ID
# and d.event_name: Optional[str] - Usually the name of the Match-Up (e.g. Bangladesh v Sri Lanka) or Race Name (e.g. R6 1400m Grp1) - Note: Dictionaries don't support wildcard searches
# and d.event_type_id: str - SportID [Soccer - 1, Tennis - 2, Golf - 3, Cricket - 4, AFL - 61420]
# and d.market_base_rate: float - Market Commission Rate
# and d.market_type: str - e.g. "MATCH_ODDS", "1ST_INNINGS_RUNS","TO_QUALIFY" - always all caps with "_" replacing spaces
# and d.name: Optional[str] - market name (e.g. Sri Lanka 1st Inns Runs)
# and d.number_of_active_runners: int - Head-To-Heads markets will be 2
# and d.number_of_winners: int - Odds Markets usually 1, Line/Handicap markets usually 0
# and d.regulators: str - 'MR_INT' to remove ring-fenced exchange markets 
# and d.turn_in_play_enabled: bool - True, False


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

def pull_ladder(availableLadder, n = 3):
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

                elif market_book.status == "CLOSED":

                    # Stop once market settled
                    break

                else:
                    
                    seconds_to_start = (market_book.market_definition.market_time - market_book.publish_time).total_seconds()

                    if seconds_to_start > log1_start:
                        
                        # Too early before off to start logging prices
                        continue

                    else:
                    
                        # Update data at different time steps depending on seconds to off
                        wait = np.where(seconds_to_start <= log2_start, log2_step, log1_step)

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
                        atb_ladder = pull_ladder(runner.ex.available_to_back, n = 3)
                        atl_ladder = pull_ladder(runner.ex.available_to_lay, n = 3)
                    except:
                        atb_ladder = {}
                        atl_ladder = {}

                    # Calculate Current Traded Volume + Traded WAP
                    limitTradedVol = sum([rung.size for rung in runner.ex.traded_volume])
                    if limitTradedVol == 0:
                        limitWAP = ""
                    else:
                        limitWAP = sum([rung.size * rung.price for rung in runner.ex.traded_volume]) / limitTradedVol
                        limitWAP = round(limitWAP, 2)

                    #Use this section to write rows that are required to join the metadata OR that will change in time
                    o.writerow(
                        (
                            market_book.market_id,
                            runner.selection_id,
                            market_book.publish_time,
                            market_book.inplay,
                            limitTradedVol,
                            limitWAP,
                            runner.last_price_traded or "",
                            str(atb_ladder).replace(' ',''), 
                            str(atl_ladder).replace(' ','')
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
        
        writer.writerow(("market_id","selection_id","time","inplay","traded_volume","wap","ltp","atb_ladder","atl_ladder"))

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
        output.write("market_id,market_time,market_type,event_name,market_name,selection_id,x,selection_name,y,result\n")
        #loop over each market in the TAR file
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
                # calculating SP traded vol as smaller of back_stake_taken or (lay_liability_taken / (BSP - 1))        
                min_gr0(
                    next((pv.size for pv in r.sp.back_stake_taken if pv.size > 0), 0),
                    next((pv.size for pv in r.sp.lay_liability_taken if pv.size > 0), 0)  / ((r.sp.actual_sp if (type(r.sp.actual_sp) is float) or (type(r.sp.actual_sp) is int) else 0) - 1)
                ) if r.sp.actual_sp is not None else 0,
            ) for r in postplay_market.runners ]
            
            # generic selection data
            for r in final_market.runners:
                selection_id=r.selection_id,
                selection_name=next((rd.name for rd in final_market.market_definition.runners if rd.selection_id == r.selection_id), None),
                selection_status=r.status
            
            # printing to csv for each selection
                output.write(
                    "{},{},{},{},{},{},{},{}\n".format(
                        postplay_market.market_id,
                        postplay_market.market_definition.market_time,
                        postplay_market.market_definition.market_type,
                        postplay_market.market_definition.event_name,
                        postplay_market.market_definition.name,
                        selection_id,
                        selection_name,
                        selection_status
                    )
                )

    #loading selection file, parsing dates and cleaning the table
    # loading selection file, parsing dates and cleaning the table
    selection = pd.read_csv(
        selection_meta, dtype={'market_id': object, 'selection_id': object}, parse_dates=['market_time']
    )
    selection.set_axis(
        [
            'market_id',
            'market_time',
            'market_type',
            'event_name',
            'market_name',
            'selection_id',
            'x',
            'selection_name',
            'y',
            'result'
        ],
        axis=1
    )

    selection = selection[['market_id','market_time','market_type','event_name','market_name','selection_id','selection_name','result']]
    selection['selection_id'] = selection['selection_id'].str.split('\(').str[1]
    selection['selection_name'] = selection['selection_name'].str.split("\('").str[1]
    selection['selection_name'] = selection['selection_name'].str.split("'").str[0]

    # loading price file and parsing dates
    prices = pd.read_csv(
        prices_path,
        quoting=csv.QUOTE_ALL,
        dtype={'market_id': 'string', 'selection_id': 'string', 'atb_ladder': 'string', 'atl_ladder': 'string'},
        parse_dates=['time']
    )

    # creating the ladder as a dictionary
    prices['atb_ladder'] = [ast.literal_eval(x) for x in prices['atb_ladder']]
    prices['atl_ladder'] = [ast.literal_eval(x) for x in prices['atl_ladder']]

    # merging the price and selection files
    df = selection.merge(prices, on=['market_id', 'selection_id'])

    # assigning best prices available and calculating time relative to market start time
    df = (
        df
        .assign(back_best=lambda x: [np.nan if d.get('p') is None else d.get('p')[0] for d in x['atb_ladder']])
        .assign(lay_best=lambda x: [np.nan if d.get('p') is None else d.get('p')[0] for d in x['atl_ladder']])
        .assign(
            seconds_before_scheduled_off=lambda x: round((x['market_time'] - x['time']).dt.total_seconds())
        )
        .query('seconds_before_scheduled_off < @log1_Start')
    )

    # Writing each processed market to its own csv file
    market_ids = df['market_id'].unique().tolist()
    for market in market_ids:
        # Create a dataframe and a naming convention for this market
        pricing_data = df[df['market_id'] == market]
        fixture = pricing_data['event_name'].iloc[0]
        market_name = pricing_data['market_name'].iloc[0]
        market_time = pricing_data['market_time'].iloc[0]
        off = market_time.strftime('%Y-%m-%d')
        # Convert GBP to AUD
        event_date = (pd.to_datetime(pricing_data['market_time']).dt.date).iloc[0]
        conversion_rate = CurrencyConverter(fallback_on_missing_rate=True).convert(1, 'GBP', 'AUD', date=event_date)
        pricing_data['traded_volume'] = pricing_data['traded_volume'] * conversion_rate
        pricing_data.loc[pricing_data['traded_volume'] < 0, 'traded_volume'] = 0
        pricing_data['traded_volume'] = pricing_data['traded_volume'].round(decimals=2)
        pricing_data.to_csv(file_directory + off + ' - ' + fixture + ' - ' + market_name + '.csv', index=False)

#removing intermediate working documents to clean up
os.remove(selection_meta)
os.remove(prices_path)