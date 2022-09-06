from joblib import load
import os
import sys

# Allow imports from src folder
module_path = os.path.abspath(os.path.join('../src'))
if module_path not in sys.path:
    sys.path.append(module_path)

from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from dateutil import tz
from pandas.tseries.offsets import MonthEnd
from sklearn.preprocessing import MinMaxScaler
import itertools

import numpy as np
import pandas as pd
from nltk.tokenize import regexp_tokenize

# settings to display all columns
pd.set_option("display.max_columns", None)

import fasttrack as ft

from dotenv import load_dotenv
load_dotenv()

# Import libraries for logging in
import betfairlightweight
from flumine import Flumine, clients
# Import libraries and logging
from flumine import BaseStrategy 
from flumine.order.trade import Trade
from flumine.order.order import LimitOrder
from flumine.markets.market import Market
from betfairlightweight.filters import streaming_market_filter
from betfairlightweight.resources import MarketBook
import re
import pandas as pd
import numpy as np
import datetime
import logging
logging.basicConfig(filename = 'how_to_automate_4.log', level=logging.INFO, format='%(asctime)s:%(levelname)s:%(message)s')

# import logging
from flumine.worker import BackgroundWorker
from flumine.events.events import TerminationEvent

import csv
from flumine.controls.loggingcontrols import LoggingControl
from flumine.order.ordertype import OrderTypes

logger = logging.getLogger(__name__)

brunos_model = load('logistic_regression.joblib')
brunos_model

# Validate FastTrack API connection
api_key = os.getenv('FAST_TRACK_API_KEY',)
client = ft.Fasttrack(api_key)
track_codes = client.listTracks()

# Import race data excluding NZ races
au_tracks_filter = list(track_codes[track_codes['state'] != 'NZ']['track_code'])

# Time window to import data
# First day of the month 46 months back from now
date_from = (datetime.today() - relativedelta(months=46)).replace(day=1).strftime('%Y-%m-%d')
# First day of previous month
date_to = (datetime.today() - relativedelta(months=1)).replace(day=1).strftime('%Y-%m-%d')

# Dataframes to populate data with
race_details = pd.DataFrame()
dog_results = pd.DataFrame()

# For each month, either fetch data from API or use local CSV file if we already have downloaded it
for start in pd.date_range(date_from, date_to, freq='MS'):
    start_date = start.strftime("%Y-%m-%d")
    end_date = (start + MonthEnd(1)).strftime("%Y-%m-%d")
    try:
        filename_races = f'FT_AU_RACES_{start_date}.csv'
        filename_dogs = f'FT_AU_DOGS_{start_date}.csv'

        filepath_races = f'../data/{filename_races}'
        filepath_dogs = f'../data/{filename_dogs}'

        print(f'Loading data from {start_date} to {end_date}')
        if os.path.isfile(filepath_races):
            # Load local CSV file
            month_race_details = pd.read_csv(filepath_races) 
            month_dog_results = pd.read_csv(filepath_dogs) 
        else:
            # Fetch data from API
            month_race_details, month_dog_results = client.getRaceResults(start_date, end_date, au_tracks_filter)
            month_race_details.to_csv(filepath_races, index=False)
            month_dog_results.to_csv(filepath_dogs, index=False)

        # Combine monthly data
        race_details = race_details.append(month_race_details, ignore_index=True)
        dog_results = dog_results.append(month_dog_results, ignore_index=True)
    except:
        print(f'Could not load data from {start_date} to {end_date}')

race_details.tail()

current_month_start_date = pd.Timestamp.now().replace(day=1).strftime("%Y-%m-%d")
current_month_end_date = (pd.Timestamp.now().replace(day=1)+ MonthEnd(1))
current_month_end_date = (current_month_end_date - pd.Timedelta('1 day')).strftime("%Y-%m-%d")

print(f'Start date: {current_month_start_date}')
print(f'End Date: {current_month_end_date}')

# Download data for races that have concluded this current month up untill today
# Start and end dates for current month
current_month_start_date = pd.Timestamp.now().replace(day=1).strftime("%Y-%m-%d")
current_month_end_date = (pd.Timestamp.now().replace(day=1)+ MonthEnd(1))
current_month_end_date = (current_month_end_date - pd.Timedelta('1 day')).strftime("%Y-%m-%d")

# Files names 
filename_races = f'FT_AU_RACES_{current_month_start_date}.csv'
filename_dogs = f'FT_AU_DOGS_{current_month_start_date}.csv'
# Where to store files locally
filepath_races = f'../data/{filename_races}'
filepath_dogs = f'../data/{filename_dogs}'

# Fetch data from API
month_race_details, month_dog_results = client.getRaceResults(current_month_start_date, current_month_end_date, au_tracks_filter)

# Save the files locally and replace any out of date fields
month_race_details.to_csv(filepath_races, index=False)
month_dog_results.to_csv(filepath_dogs, index=False)

dog_results

# This is super important I have spent literally hours before I found out this was causing errors
dog_results['@id'] = pd.to_numeric(dog_results['@id'])

# Append the extra data to our data frames 
race_details = race_details.append(month_race_details, ignore_index=True)
dog_results = dog_results.append(month_dog_results, ignore_index=True)

# Download the data for todays races
todays_date = pd.Timestamp.now().strftime("%Y-%m-%d")
todays_races, todays_dogs = client.getFullFormat(dt= todays_date, tracks = au_tracks_filter)

# display is for ipython notebooks only
# display(todays_races.head(1), todays_dogs.head(1))

# It seems that the todays_races dataframe doesn't have the date column, so let's add that on
todays_races['date'] = pd.Timestamp.now().strftime('%d %b %y')
todays_races.head(1)

# It also seems that in todays_dogs dataframe Box is labeled as RaceBox instead, so let's rename it
# We can also see that there are some specific dogs that have "Res." as a suffix of their name, i.e. they are reserve dogs,
# We will treat this later
todays_dogs = todays_dogs.rename(columns={"RaceBox":"Box"})
todays_dogs.tail(3)

# Appending todays data to this months data
month_dog_results = pd.concat([month_dog_results,todays_dogs],join='outer')[month_dog_results.columns]
month_race_details = pd.concat([month_race_details,todays_races],join='outer')[month_race_details.columns]

# Appending this months data to the rest of our historical data
race_details = race_details.append(month_race_details, ignore_index=True)
dog_results = dog_results.append(month_dog_results, ignore_index=True)

race_details

## Cleanse and normalise the data
# Clean up the race dataset
race_details = race_details.rename(columns = {'@id': 'FastTrack_RaceId'})
race_details['Distance'] = race_details['Distance'].apply(lambda x: int(x.replace("m", "")))
race_details['date_dt'] = pd.to_datetime(race_details['date'], format = '%d %b %y')
# Clean up the dogs results dataset
dog_results = dog_results.rename(columns = {'@id': 'FastTrack_DogId', 'RaceId': 'FastTrack_RaceId'})

# New line of code (rest of this code chunk is copied from bruno's code)
dog_results['FastTrack_DogId'] = pd.to_numeric(dog_results['FastTrack_DogId'])

# Combine dogs results with race attributes
dog_results = dog_results.merge(
    race_details, 
    how = 'left',
    on = 'FastTrack_RaceId'
)

# Convert StartPrice to probability
dog_results['StartPrice'] = dog_results['StartPrice'].apply(lambda x: None if x is None else float(x.replace('$', '').replace('F', '')) if isinstance(x, str) else x)
dog_results['StartPrice_probability'] = (1 / dog_results['StartPrice']).fillna(0)
dog_results['StartPrice_probability'] = dog_results.groupby('FastTrack_RaceId')['StartPrice_probability'].apply(lambda x: x / x.sum())

# Discard entries without results (scratched or did not finish)
dog_results = dog_results[~dog_results['Box'].isnull()]
dog_results['Box'] = dog_results['Box'].astype(int)

# Clean up other attributes
dog_results['RunTime'] = dog_results['RunTime'].astype(float)
dog_results['SplitMargin'] = dog_results['SplitMargin'].astype(float)
dog_results['Prizemoney'] = dog_results['Prizemoney'].astype(float).fillna(0)
dog_results['Place'] = pd.to_numeric(dog_results['Place'].apply(lambda x: x.replace("=", "") if isinstance(x, str) else 0), errors='coerce').fillna(0)
dog_results['win'] = dog_results['Place'].apply(lambda x: 1 if x == 1 else 0)

# Normalise some of the raw values
dog_results['Prizemoney_norm'] = np.log10(dog_results['Prizemoney'] + 1) / 12
dog_results['Place_inv'] = (1 / dog_results['Place']).fillna(0)
dog_results['Place_log'] = np.log10(dog_results['Place'] + 1).fillna(0)
dog_results['RunSpeed'] = (dog_results['RunTime'] / dog_results['Distance']).fillna(0)

## Generate features using raw data
# Calculate median winner time per track/distance
win_results = dog_results[dog_results['win'] == 1]
median_win_time = pd.DataFrame(data=win_results[win_results['RunTime'] > 0].groupby(['Track', 'Distance'])['RunTime'].median()).rename(columns={"RunTime": "RunTime_median"}).reset_index()
median_win_split_time = pd.DataFrame(data=win_results[win_results['SplitMargin'] > 0].groupby(['Track', 'Distance'])['SplitMargin'].median()).rename(columns={"SplitMargin": "SplitMargin_median"}).reset_index()
median_win_time.head()

# Calculate track speed index
median_win_time['speed_index'] = (median_win_time['RunTime_median'] / median_win_time['Distance'])
median_win_time['speed_index'] = MinMaxScaler().fit_transform(median_win_time[['speed_index']])
median_win_time.head()

# Compare dogs finish time with median winner time
dog_results = dog_results.merge(median_win_time, on=['Track', 'Distance'], how='left')
dog_results = dog_results.merge(median_win_split_time, on=['Track', 'Distance'], how='left')

# Normalise time comparison
dog_results['RunTime_norm'] = (dog_results['RunTime_median'] / dog_results['RunTime']).clip(0.9, 1.1)
dog_results['RunTime_norm'] = MinMaxScaler().fit_transform(dog_results[['RunTime_norm']])
dog_results['SplitMargin_norm'] = (dog_results['SplitMargin_median'] / dog_results['SplitMargin']).clip(0.9, 1.1)
dog_results['SplitMargin_norm'] = MinMaxScaler().fit_transform(dog_results[['SplitMargin_norm']])
dog_results.head()

# Calculate box winning percentage for each track/distance
box_win_percent = pd.DataFrame(data=dog_results.groupby(['Track', 'Distance', 'Box'])['win'].mean()).rename(columns={"win": "box_win_percent"}).reset_index()
# Add to dog results dataframe
dog_results = dog_results.merge(box_win_percent, on=['Track', 'Distance', 'Box'], how='left')
# Display example of barrier winning probabilities
print(box_win_percent.head(8))

dog_results[dog_results['FastTrack_DogId'] == 592253143].tail()[['date_dt','Place','DogName','RaceNum','Track','Distance','win','Prizemoney_norm','Place_inv','Place_log']]

# Generate rolling window features
dataset = dog_results.copy()
dataset = dataset.set_index(['FastTrack_DogId', 'date_dt']).sort_index()

# Use rolling window of 28, 91 and 365 days
rolling_windows = ['28D', '91D', '365D']
# Features to use for rolling windows calculation
features = ['RunTime_norm', 'SplitMargin_norm', 'Place_inv', 'Place_log', 'Prizemoney_norm']
# Aggregation functions to apply
aggregates = ['min', 'max', 'mean', 'median', 'std']
# Keep track of generated feature names
feature_cols = ['speed_index', 'box_win_percent']

for rolling_window in rolling_windows:
        print(f'Processing rolling window {rolling_window}')

        rolling_result = (
            dataset
            .reset_index(level=0).sort_index()
            .groupby('FastTrack_DogId')[features]
            .rolling(rolling_window)
            .agg(aggregates)
            .groupby(level=0)  # Thanks to Brett for finding this!
            .shift(1)
        )

        # My own dodgey code to work with reserve dogs
        temp = rolling_result.reset_index()
        temp = temp[temp['date_dt'] == pd.Timestamp.now().normalize()]
        temp.groupby(['FastTrack_DogId','date_dt']).first()
        rolling_result.loc[pd.IndexSlice[:, pd.Timestamp.now().normalize()], :] = temp.groupby(['FastTrack_DogId','date_dt']).first()

        # Generate list of rolling window feature names (eg: RunTime_norm_min_365D)
        agg_features_cols = [f'{f}_{a}_{rolling_window}' for f, a in itertools.product(features, aggregates)]
        # Add features to dataset
        dataset[agg_features_cols] = rolling_result
        # Keep track of generated feature names
        feature_cols.extend(agg_features_cols)

# Replace missing values with 0
dataset.fillna(0, inplace=True)
# display(dataset.head(8))  # display is only for ipython notebooks

# Only keep data after 2018-12-01
model_df = dataset.reset_index()
feature_cols = np.unique(feature_cols).tolist()
model_df = model_df[model_df['date_dt'] >= '2018-12-01']

# This line was originally part of Bruno's tutorial, but we don't run it in this script
# model_df = model_df[['date_dt', 'FastTrack_RaceId', 'DogName', 'win', 'StartPrice_probability'] + feature_cols]

# Only train model off of races where each dog has a value for each feature
races_exclude = model_df[model_df.isnull().any(axis = 1)]['FastTrack_RaceId'].drop_duplicates()
model_df = model_df[~model_df['FastTrack_RaceId'].isin(races_exclude)]

# Generate predictions like normal
# Range of dates that we want to simulate later '2022-03-01' to '2022-04-01'
todays_data = model_df[(model_df['date_dt'] >= pd.Timestamp('2022-03-01').strftime('%Y-%m-%d')) & (model_df['date_dt'] < pd.Timestamp('2022-04-01').strftime('%Y-%m-%d'))]
dog_win_probabilities = brunos_model.predict_proba(todays_data[feature_cols])[:,1]
todays_data['prob_LogisticRegression'] = dog_win_probabilities
todays_data['renormalise_prob'] = todays_data.groupby('FastTrack_RaceId')['prob_LogisticRegression'].apply(lambda x: x / x.sum())
todays_data['rating'] = 1/todays_data['renormalise_prob']
todays_data = todays_data.sort_values(by = 'date_dt')
todays_data

def download_iggy_ratings(date):
    """Downloads the Betfair Iggy model ratings for a given date and formats it into a nice DataFrame.

    Args:
        date (datetime): the date we want to download the ratings for
    """
    iggy_url_1 = 'https://betfair-data-supplier-prod.herokuapp.com/api/widgets/iggy-joey/datasets?date='
    iggy_url_2 = date.strftime("%Y-%m-%d")
    iggy_url_3 = '&presenter=RatingsPresenter&csv=true'
    iggy_url = iggy_url_1 + iggy_url_2 + iggy_url_3

    # Download todays greyhounds ratings
    iggy_df = pd.read_csv(iggy_url)

    # Data clearning
    iggy_df = iggy_df.rename(
    columns={
        "meetings.races.bfExchangeMarketId":"market_id",
        "meetings.races.runners.bfExchangeSelectionId":"selection_id",
        "meetings.races.runners.ratedPrice":"rating",
        "meetings.races.number":"RaceNum",
        "meetings.name":"Track",
        "meetings.races.runners.name":"DogName"
        }
    )
    # iggy_df = iggy_df[['market_id','selection_id','rating']]
    iggy_df['market_id'] = iggy_df['market_id'].astype(str)
    iggy_df['date_dt'] = date

    # Set market_id and selection_id as index for easy referencing
    # iggy_df = iggy_df.set_index(['market_id','selection_id'])
    return(iggy_df)

# Download historical ratings over a time period and convert into a big DataFrame.
back_test_period = pd.date_range(start='2022-03-01', end='2022-04-01')
frames = [download_iggy_ratings(day) for day in back_test_period]
iggy_df = pd.concat(frames)
iggy_df

# format DogNames to merge
todays_data['DogName'] = todays_data['DogName'].apply(lambda x: x.replace("'", "").replace(".", "").replace("Res", "").strip())
iggy_df['DogName'] = iggy_df['DogName'].str.upper()
# Merge
backtest = iggy_df[['market_id','selection_id','DogName','date_dt']].merge(todays_data[['rating','DogName','date_dt']], how = 'inner', on = ['DogName','date_dt'])
backtest

# Save predictions for if we want to backtest/simulate it later
backtest.to_csv('backtest.csv', index=False) # Csv format
# backtest.to_pickle('backtest.pkl') # pickle format (faster, but can't open in excel)

todays_data[todays_data['FastTrack_RaceId'] == '798906744']

# Select todays data 
todays_data = model_df[model_df['date_dt'] == pd.Timestamp.now().strftime('%Y-%m-%d')]

# Generate runner win predictions
dog_win_probabilities = brunos_model.predict_proba(todays_data[feature_cols])[:,1]
todays_data['prob_LogisticRegression'] = dog_win_probabilities

# We no longer renomralise probability in this chunk of code, do it in Flumine instead
# todays_data['renormalise_prob'] = todays_data.groupby('FastTrack_RaceId')['prob_LogisticRegression'].apply(lambda x: x / x.sum())
# todays_data['rating'] = 1/todays_data['renormalise_prob']
# todays_data = todays_data.sort_values(by = 'date_dt')

todays_data

# Prepare data for easy reference in flumine
todays_data['DogName_bf'] = todays_data['DogName'].apply(lambda x: x.replace("'", "").replace(".", "").replace("Res", "").strip())
todays_data.replace({'Sandown (SAP)': 'Sandown Park'}, regex=True, inplace=True)
todays_data = todays_data.set_index(['DogName_bf','Track','RaceNum'])
todays_data.head()

# Credentials to login and logging in 
trading = betfairlightweight.APIClient('username','password',app_key='appkey')
client = clients.BetfairClient(trading, interactive_login=True)

# Login
framework = Flumine(client=client)

# Code to login when using security certificates
# trading = betfairlightweight.APIClient('username','password',app_key='appkey', certs=r'C:\Users\zhoui\openssl_certs')
# client = clients.BetfairClient(trading)

# framework = Flumine(client=client)

class FlatBetting(BaseStrategy):
    def start(self) -> None:
        print("starting strategy 'FlatBetting' using the model we created the Greyhound modelling in Python Tutorial")

    def check_market_book(self, market: Market, market_book: MarketBook) -> bool:
        if market_book.status != "CLOSED":
            return True

    def process_market_book(self, market: Market, market_book: MarketBook) -> None:
        # Convert dataframe to a global variable
        global todays_data

        # At the 60 second mark:
        if market.seconds_to_start < 60 and market_book.inplay == False:
            # get the list of dog_names, name of the track/venue and race_number/RaceNum from Betfair Polling API
            dog_names = []
            track = market.market_catalogue.event.venue
            race_number = market.market_catalogue.market_name.split(' ',1)[0]  # comes out as R1/R2/R3 .. etc
            race_number = re.sub("[^0-9]", "", race_number)  # only keep the numbers 
            for runner_cata in market.market_catalogue.runners:
                dog_name = runner_cata.runner_name.split(' ',1)[1].upper()
                dog_names.append(dog_name)

            # Check if there are box changes, if there are then use Brett's code
            if market.market_catalogue.description.clarifications != None:
                # Brett's code to get Box changes:
                my_string = market.market_catalogue.description.clarifications.replace("<br> Dog","<br>Dog")
                pattern1 = r'(?<=<br>Dog ).+?(?= starts)'
                pattern2 = r"(?<=\bbox no. )(\w+)"
                runners_df = pd.DataFrame (regexp_tokenize(my_string, pattern1), columns = ['runner_name'])
                runners_df['runner_name'] = runners_df['runner_name'].astype(str)
                # Remove dog name from runner_number
                runners_df['runner_number'] = runners_df['runner_name'].apply(lambda x: x[:(x.find(" ") - 1)].upper())
                # Remove dog number from runner_name
                runners_df['runner_name'] = runners_df['runner_name'].apply(lambda x: x[(x.find(" ") + 1):].upper())
                runners_df['Box'] = regexp_tokenize(my_string, pattern2)

                # Replace any old Box info in our original dataframe with data available in runners_df
                runners_df = runners_df.set_index('runner_name')
                todays_data.loc[(runners_df.index[runners_df.index.isin(dog_names)],track,race_number),'Box'] = runners_df.loc[runners_df.index.isin(dog_names),'Box'].to_list()
                # Merge box_win_percentage back on:
                todays_data = todays_data.drop(columns = 'box_win_percentage', axis = 1)
                todays_data = todays_data.reset_index().merge(box_win_percent, on = ['Track', 'Distance','Box'], how = 'left').set_index(['DogName_bf','Track','RaceNum'])

            # Generate probabilities using Bruno's model
            todays_data.loc[(dog_names,track,race_number),'prob_LogisticRegression'] = brunos_model.predict_proba(todays_data.loc[(dog_names,track,race_number)][feature_cols])[:,1]
            # renomalise probabilities
            probabilities = todays_data.loc[dog_names,track,race_number]['prob_LogisticRegression']
            todays_data.loc[(dog_names,track,race_number),'renormalised_prob'] = probabilities/probabilities.sum()
            # convert probaiblities to ratings
            todays_data.loc[(dog_names,track,race_number),'rating'] = 1/todays_data.loc[dog_names,track,race_number]['renormalised_prob']

            # Use both the polling api (market.catalogue) and the streaming api at once:
            for runner_cata, runner in zip(market.market_catalogue.runners, market_book.runners):
                # Check the polling api and streaming api matches up (sometimes it doesn't)
                if runner_cata.selection_id == runner.selection_id:
                    # Get the dog_name from polling api then reference our data for our model rating
                    dog_name = runner_cata.runner_name.split(' ',1)[1].upper()
                    
                    # Rest is the same as How to Automate III
                    model_price = todays_data.loc[dog_name,track,race_number]['rating']
                    ### If you have an issue such as:
                        # Unknown error The truth value of a Series is ambiguous. Use a.empty, a.bool(), a.item(), a.any() or a.all().
                        # Then do model_price = todays_data.loc[dog_name,track,race_number]['rating'].item()

                    # Log info before placing bets
                    logging.info(f'dog_name: {dog_name}')
                    logging.info(f'model_price: {model_price}')
                    logging.info(f'market_id: {market_book.market_id}')
                    logging.info(f'selection_id: {runner.selection_id}')
                    
                    # If best available to back price is > rated price then flat $5 back
                    if runner.status == "ACTIVE" and runner.ex.available_to_back[0]['price'] > model_price:
                        trade = Trade(
                        market_id=market_book.market_id,
                        selection_id=runner.selection_id,
                        handicap=runner.handicap,
                        strategy=self,
                        )
                        order = trade.create_order(
                            side="BACK", order_type=LimitOrder(price=runner.ex.available_to_back[0]['price'], size=5.00)
                        )
                        market.place_order(order)
                    # If best available to lay price is < rated price then flat $5 lay
                    if runner.status == "ACTIVE" and runner.ex.available_to_lay[0]['price'] < model_price:
                        trade = Trade(
                        market_id=market_book.market_id,
                        selection_id=runner.selection_id,
                        handicap=runner.handicap,
                        strategy=self,
                        )
                        order = trade.create_order(
                            side="LAY", order_type=LimitOrder(price=runner.ex.available_to_lay[0]['price'], size=5.00)
                        )
                        market.place_order(order)

greyhounds_strategy = FlatBetting(
    market_filter=streaming_market_filter(
        event_type_ids=["4339"], # Greyhounds markets
        country_codes=["AU"], # Australian markets
        market_types=["WIN"], # Win markets
    ),
    max_order_exposure= 50, # Max exposure per order = 50
    max_trade_count=1, # Max 1 trade per selection
    max_live_trade_count=1, # Max 1 unmatched trade per selection
)

framework.add_strategy(greyhounds_strategy)

# logger = logging.getLogger(__name__)

"""
Worker can be used as followed:
    framework.add_worker(
        BackgroundWorker(
            framework,
            terminate,
            func_kwargs={"today_only": True, "seconds_closed": 1200},
            interval=60,
            start_delay=60,
        )
    )
This will run every 60s and will terminate 
the framework if all markets starting 'today' 
have been closed for at least 1200s
"""


# Function that stops automation running at the end of the day
def terminate(
    context: dict, flumine, today_only: bool = True, seconds_closed: int = 600
) -> None:
    """terminate framework if no markets
    live today.
    """
    markets = list(flumine.markets.markets.values())
    markets_today = [
        m
        for m in markets
        if m.market_start_datetime.date() == datetime.datetime.utcnow().date()
        and (
            m.elapsed_seconds_closed is None
            or (m.elapsed_seconds_closed and m.elapsed_seconds_closed < seconds_closed)
        )
    ]
    if today_only:
        market_count = len(markets_today)
    else:
        market_count = len(markets)
    if market_count == 0:
        # logger.info("No more markets available, terminating framework")
        flumine.handler_queue.put(TerminationEvent(flumine))

# Add the stopped to our framework
framework.add_worker(
    BackgroundWorker(
        framework,
        terminate,
        func_kwargs={"today_only": True, "seconds_closed": 1200},
        interval=60,
        start_delay=60,
    )
)

logger = logging.getLogger(__name__)

FIELDNAMES = [
    "bet_id",
    "strategy_name",
    "market_id",
    "selection_id",
    "trade_id",
    "date_time_placed",
    "price",
    "price_matched",
    "size",
    "size_matched",
    "profit",
    "side",
    "elapsed_seconds_executable",
    "order_status",
    "market_note",
    "trade_notes",
    "order_notes",
]


class LiveLoggingControl(LoggingControl):
    NAME = "BACKTEST_LOGGING_CONTROL"

    def __init__(self, *args, **kwargs):
        super(LiveLoggingControl, self).__init__(*args, **kwargs)
        self._setup()

    # Changed file path and checks if the file orders_hta_4.csv already exists, if it doens't then create it
    def _setup(self):
        if os.path.exists("orders_hta_4.csv"):
            logging.info("Results file exists")
        else:
            with open("orders_hta_4.csv", "w") as m:
                csv_writer = csv.DictWriter(m, delimiter=",", fieldnames=FIELDNAMES)
                csv_writer.writeheader()

    def _process_cleared_orders_meta(self, event):
        orders = event.event
        with open("orders_hta_4.csv", "a") as m:
            for order in orders:
                if order.order_type.ORDER_TYPE == OrderTypes.LIMIT:
                    size = order.order_type.size
                else:
                    size = order.order_type.liability
                if order.order_type.ORDER_TYPE == OrderTypes.MARKET_ON_CLOSE:
                    price = None
                else:
                    price = order.order_type.price
                try:
                    order_data = {
                        "bet_id": order.bet_id,
                        "strategy_name": order.trade.strategy,
                        "market_id": order.market_id,
                        "selection_id": order.selection_id,
                        "trade_id": order.trade.id,
                        "date_time_placed": order.responses.date_time_placed,
                        "price": price,
                        "price_matched": order.average_price_matched,
                        "size": size,
                        "size_matched": order.size_matched,
                        "profit": 0 if not order.cleared_order else order.cleared_order.profit,
                        "side": order.side,
                        "elapsed_seconds_executable": order.elapsed_seconds_executable,
                        "order_status": order.status.value,
                        "market_note": order.trade.market_notes,
                        "trade_notes": order.trade.notes_str,
                        "order_notes": order.notes_str,
                    }
                    csv_writer = csv.DictWriter(m, delimiter=",", fieldnames=FIELDNAMES)
                    csv_writer.writerow(order_data)
                except Exception as e:
                    logger.error(
                        "_process_cleared_orders_meta: %s" % e,
                        extra={"order": order, "error": e},
                    )

        logger.info("Orders updated", extra={"order_count": len(orders)})

    def _process_cleared_markets(self, event):
        cleared_markets = event.event
        for cleared_market in cleared_markets.orders:
            logger.info(
                "Cleared market",
                extra={
                    "market_id": cleared_market.market_id,
                    "bet_count": cleared_market.bet_count,
                    "profit": cleared_market.profit,
                    "commission": cleared_market.commission,
                },
            )

framework.add_logging_control(
    LiveLoggingControl()
)

framework.run()