# Import libraries
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

import math
import numpy as np
import pandas as pd
import fasttrack as ft

from dotenv import load_dotenv
load_dotenv()

# Validate FastTrack API connection
api_key = os.getenv('FAST_TRACK_API_KEY', '<REPLACE WITH YOUR KEY>')
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

## Cleanse and normalise the data
# Clean up the race dataset
race_details = race_details.rename(columns = {'@id': 'FastTrack_RaceId'})
race_details['Distance'] = race_details['Distance'].apply(lambda x: int(x.replace("m", "")))
race_details['date_dt'] = pd.to_datetime(race_details['date'], format = '%d %b %y')
# Clean up the dogs results dataset
dog_results = dog_results.rename(columns = {'@id': 'FastTrack_DogId', 'RaceId': 'FastTrack_RaceId'})

# Combine dogs results with race attributes
dog_results = dog_results.merge(
    race_details[['FastTrack_RaceId', 'Distance', 'RaceGrade', 'Track', 'date_dt']], 
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
display(box_win_percent.head(8))

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
            .reset_index(level=0)
            .groupby('FastTrack_DogId')[features]
            .rolling(rolling_window)
            .agg(aggregates)
            .shift(1)
        )

        # Generate list of rolling window feature names (eg: RunTime_norm_min_365D)
        agg_features_cols = [f'{f}_{a}_{rolling_window}' for f, a in itertools.product(features, aggregates)]
        # Add features to dataset
        dataset[agg_features_cols] = rolling_result
        # Keep track of generated feature names
        feature_cols.extend(agg_features_cols)

# Replace missing values with 0
dataset.fillna(0, inplace=True)
display(dataset.head(8))

# Only keep data after 2018-12-01
model_df = dataset.reset_index()
feature_cols = np.unique(feature_cols).tolist()
model_df = model_df[model_df['date_dt'] >= '2018-12-01']
model_df = model_df[['date_dt', 'FastTrack_RaceId', 'DogName', 'win', 'StartPrice_probability'] + feature_cols]

# Only train model off of races where each dog has a value for each feature
races_exclude = model_df[model_df.isnull().any(axis = 1)]['FastTrack_RaceId'].drop_duplicates()
model_df = model_df[~model_df['FastTrack_RaceId'].isin(races_exclude)]

## Build and train Regression models
from matplotlib import pyplot
from matplotlib.pyplot import figure

from sklearn.linear_model import LogisticRegression

# Split the data into train and test data
train_data = model_df[model_df['date_dt'] < '2021-01-01'].reset_index(drop = True).sample(frac=1)
test_data = model_df[model_df['date_dt'] >= '2021-01-01'].reset_index(drop = True)

# Use our previously built features set columns as Training vector
# Use win flag as Target vector
train_x, train_y = train_data[feature_cols], train_data['win']
test_x, test_y = test_data[feature_cols], test_data['win']

# Build a LogisticRegression model
model = LogisticRegression(verbose=0, solver='saga', n_jobs=-1)

# Train the model
print(f'Training on {len(train_x):,} samples with {len(feature_cols)} features')
model.fit(train_x, train_y)

# Generate runner win predictions
dog_win_probabilities = model.predict_proba(test_x)[:,1]
test_data['prob_LogisticRegression'] = dog_win_probabilities
# Normalise probabilities
test_data['prob_LogisticRegression'] = test_data.groupby('FastTrack_RaceId')['prob_LogisticRegression'].apply(lambda x: x / sum(x))

# Create a boolean column for whether a dog has the higehst model prediction in a race
test_dataset_size = test_data['FastTrack_RaceId'].nunique()
odds_win_prediction = test_data.groupby('FastTrack_RaceId')['prob_LogisticRegression'].apply(lambda x: x == max(x))
odds_win_prediction_percent = len(test_data[(odds_win_prediction == True) & (test_data['win'] == 1)]) / test_dataset_size
print(f"LogisticRegression strike rate: {odds_win_prediction_percent:.2%}")

from sklearn.metrics import brier_score_loss

brier_score = brier_score_loss(test_data['win'], test_data['prob_LogisticRegression'])
print(f'LogisticRegression Brier score: {brier_score:.8}')

# Predictions distribution
import matplotlib.pyplot as plt
import seaborn as sns

bins = 100

sns.displot(data=[test_data['prob_LogisticRegression'], test_data['StartPrice_probability']], kind="hist",
             bins=bins, height=7, aspect=2)
plt.title('StartPrice vs LogisticRegression probabilities distribution')
plt.xlabel('Probability')
plt.show()

# Predictions calibration
from sklearn.calibration import calibration_curve

bins = 100
fig = plt.figure(figsize=(12, 9))

# Generate calibration curves based on our probabilities
cal_y, cal_x = calibration_curve(test_data['win'], test_data['prob_LogisticRegression'], n_bins=bins)

# Plot against reference line
plt.plot(cal_x, cal_y, marker='o', linewidth=1)
plt.plot([0, 1], [0, 1], '--', color='gray')
plt.title("LogisticRegression calibration curve");

# Other classification models
from matplotlib import pyplot
from matplotlib.pyplot import figure

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier

# Gradient Boosting Machines libraries
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

# Common models parameters
verbose       = 0
learning_rate = 0.1
n_estimators  = 100

# Train different types of models
models = {
    'LogisticRegression':         LogisticRegression(verbose=0, solver='saga', n_jobs=-1),
    'GradientBoostingClassifier': GradientBoostingClassifier(verbose=verbose, learning_rate=learning_rate, n_estimators=n_estimators, max_depth=3, max_features=0.25),
    'RandomForestClassifier':     RandomForestClassifier(verbose=verbose, n_estimators=n_estimators, max_depth=8, max_features=0.5, n_jobs=-1),
    'LGBMClassifier':             LGBMClassifier(verbose=verbose, learning_rate=learning_rate, n_estimators=n_estimators, force_col_wise=True),
    'XGBClassifier':              XGBClassifier(verbosity=verbose, learning_rate=learning_rate, n_estimators=n_estimators, objective='binary:logistic', use_label_encoder=False),
    'CatBoostClassifier':         CatBoostClassifier(verbose=verbose, learning_rate=learning_rate, n_estimators=n_estimators)
}

print(f'Training on {len(train_x):,} samples with {len(feature_cols)} features')
for key, model in models.items():
    print(f'Fitting model {key}')
    model.fit(train_x, train_y)

# Calculate probabilities for each model on the test dataset
probs_columns = ['StartPrice_probability']
for key, model in models.items():
    probs_column_key = f'prob_{key}'
    # Calculate runner win probability
    dog_win_probs = model.predict_proba(test_x)[:,1]
    test_data[probs_column_key] = dog_win_probs
    # Normalise probabilities
    test_data[probs_column_key] = test_data.groupby('FastTrack_RaceId')[f'prob_{key}'].apply(lambda x: x / sum(x))
    probs_columns.append(probs_column_key)

# Calculate model strike rate and Brier score across models
# Create a boolean column for whether a dog has the higehst model prediction in a race.
# Do the same for the starting price as a comparison
test_dataset_size = test_data['FastTrack_RaceId'].nunique()
odds_win_prediction = test_data.groupby('FastTrack_RaceId')['StartPrice_probability'].apply(lambda x: x == max(x))
odds_win_prediction_percent = len(test_data[(odds_win_prediction == True) & (test_data['win'] == 1)]) / test_dataset_size
brier_score = brier_score_loss(test_data['win'], test_data['StartPrice_probability'])
print(f'Starting Price                strike rate: {odds_win_prediction_percent:.2%} Brier score: {brier_score:.8}')

for key, model in models.items():
    predicted_winners = test_data.groupby('FastTrack_RaceId')[f'prob_{key}'].apply(lambda x: x == max(x))
    strike_rate = len(test_data[(predicted_winners == True) & (test_data['win'] == 1)]) / test_data['FastTrack_RaceId'].nunique()
    brier_score = brier_score_loss(test_data['win'], test_data[f'prob_{key}'])
    print(f'{key.ljust(30)}strike rate: {strike_rate:.2%} Brier score: {brier_score:.8}')

# Visualise model predictions
# Display and format sample results
def highlight_max(s, props=''):
    return np.where(s == np.nanmax(s.values), props, '')
def highlight_min(s, props=''):
    return np.where(s == np.nanmin(s.values), props, '')

test_data[probs_columns].sample(20).style \
    .bar(color='#FFA07A', vmin=0.01, vmax=0.25, axis=1) \
    .apply(highlight_max, props='color:red;', axis=1) \
    .apply(highlight_min, props='color:blue;', axis=1)

## Display models feature importance
from sklearn.preprocessing import normalize

total_feature_importances = []

# Individual models feature importance
for key, model in models.items():
    figure(figsize=(10, 24), dpi=80)
    if isinstance(model, LogisticRegression):
        feature_importance = model.coef_[0]
    else:
        feature_importance = model.feature_importances_
    
    feature_importance = normalize(feature_importance[:,np.newaxis], axis=0).ravel()
    total_feature_importances.append(feature_importance)
    pyplot.barh(feature_cols, feature_importance)
    pyplot.xlabel(f'{key} Features Importance')
    pyplot.show()

# Overall feature importance
avg_feature_importances = np.asarray(total_feature_importances).mean(axis=0)
figure(figsize=(10, 24), dpi=80)
pyplot.barh(feature_cols, avg_feature_importances)
pyplot.xlabel('Overall Features Importance')
pyplot.show()