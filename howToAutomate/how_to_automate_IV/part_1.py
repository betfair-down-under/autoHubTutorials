"""
    Data ingestion script, prepares data for cleaning and feature creation
    Downloads/reads in historic data (data from previous 46 months)
    Downloads data for todays races and combines with historic data
    Saves data as two CSVs:
        - dog_results.csv
        - race_details.csv
"""

import os
import sys
import pandas as pd
from datetime import datetime
from dateutil.relativedelta import relativedelta

# Allow imports from src folder
module_path = os.path.abspath(os.path.join('src'))
if module_path not in sys.path:
    sys.path.append(module_path)

import fasttrack as ft
from dotenv import load_dotenv
load_dotenv()

# Validate FastTrack API connection
api_key = os.getenv('FAST_TRACK_API_KEY',)
client = ft.Fasttrack(api_key)
track_codes = client.listTracks()

# Import race data excluding NZ races
au_tracks_filter = list(track_codes[track_codes['state'] != 'NZ']['track_code'])

# Time window to import data
# First day of the month 46 months back from now
date_from = (datetime.today() - relativedelta(months=46)).replace(day=1).strftime('%Y-%m-%d')
# Download historic data up untill yesterday
date_to = (datetime.today() - relativedelta(days=1)).strftime('%Y-%m-%d')

# List to populate data with, convert to dataframe after fully populated
race_details = []
dog_results = []

# Download/load historic (up until yesterday) data
# For each day, either fetch data from API or use local CSV file if we already have downloaded it
for start in pd.date_range(date_from, date_to, freq='d'):
    start_date = start.strftime("%Y-%m-%d")
    end_date = start_date
    try:
        filename_races = f'FT_AU_RACES_{start_date}.csv'
        filename_dogs = f'FT_AU_DOGS_{start_date}.csv'

        filepath_races = f'../data/{filename_races}'
        filepath_dogs = f'../data/{filename_dogs}'

        print(f'Loading data from {start_date} to {end_date}')
        if os.path.isfile(filepath_races):
            # Load local CSV file
            day_race_details = pd.read_csv(filepath_races) 
            day_dog_results = pd.read_csv(filepath_dogs) 
        else:
            # Fetch data from API
            day_race_details, day_dog_results = client.getRaceResults(start_date, end_date, au_tracks_filter)
            day_race_details.to_csv(filepath_races, index=False)
            day_dog_results.to_csv(filepath_dogs, index=False)

        # Combine daily data
        race_details.append(day_race_details)
        dog_results.append(day_dog_results)
    except:
        print(f'Could not load data from {start_date} to {end_date}')

# Download todays data from the api
todays_date = pd.Timestamp.now().strftime('%Y-%m-%d')
todays_race_details, todays_dog_results = client.getFullFormat(todays_date)

# Make live API data in the same form as historic data
todays_race_details = todays_race_details.rename(columns={"Date":"date"})
todays_dog_results = todays_dog_results.rename(columns={"RaceBox":"Box"})
# Only keep the columns we have in the historic data
usecols_race_details = ['@id','RaceNum','RaceName','RaceTime','Distance','RaceGrade','Track','date']
usecols_dog_results = ['@id','DogName','Box','RaceId','TrainerId','TrainerName']
# dog_results columns not in live data ['Place','Rug','Weight','StartPrice','Handicap','Margin1', 'Margin2','PIR','Checks','Comments','SplitMargin', 'RunTime', 'Prizemoney',]
todays_race_details = todays_race_details[usecols_race_details]
todays_dog_results = todays_dog_results[usecols_dog_results]

# Now that todays data looks similar to our historic data lets add todays data to the rest of our historic data
race_details.append(todays_race_details)
dog_results.append(todays_dog_results)
# Convert our data into a nice DataFrame
race_details = pd.concat(race_details)
dog_results = pd.concat(dog_results)

# Save our data to csv files
race_details.to_csv('../data/race_details.csv', index = False)
dog_results.to_csv('../data/dog_results.csv', index = False)
# Ready for data cleaning and feature creation
