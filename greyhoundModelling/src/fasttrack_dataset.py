import os
import sys

from datetime import datetime, timedelta
from dateutil import tz
from pandas.tseries.offsets import MonthEnd

import math
import numpy as np
import pandas as pd
import fasttrack as ft

from dotenv import load_dotenv
load_dotenv()

# Return FastTrack datasets
def load(date_from='2017-12-01', date_to='2021-07-31', exclude_states=[], inlude_states=[]):
    # Validate FastTrack API connection
    client = ft.Fasttrack(os.getenv('FAST_TRACK_API_KEY'))
    track_codes = client.listTracks()

    # States filter
    if exclude_states:
        track_codes = track_codes[~track_codes['state'].isin(exclude_states)]
    if inlude_states:
        track_codes = track_codes[track_codes['state'].isin(inlude_states)]
    tracks_filter = list(track_codes['track_code'])

    race_details = pd.DataFrame()
    dog_results = pd.DataFrame()

    # For each month of data, either use local CSV file or fetch from API
    for start in pd.date_range(date_from, date_to, freq='MS'):
        start_date = start.strftime("%Y-%m-%d")
        end_date = (start + MonthEnd(1)).strftime("%Y-%m-%d")
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
            month_race_details, month_dog_results = client.getRaceResults(start_date, end_date, tracks_filter)
            month_race_details.to_csv(filepath_races, index=False)
            month_dog_results.to_csv(filepath_dogs, index=False)

        # Combine monthly data
        race_details = race_details.append(month_race_details, ignore_index=True)
        dog_results = dog_results.append(month_dog_results, ignore_index=True)

    return race_details, dog_results
