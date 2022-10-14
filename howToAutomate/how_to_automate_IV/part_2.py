"""
    Script reads in data downloaded from part_1 (data ingestion)
    Cleans and normalises data
    Create our features
    Prepares dataframe for part_3 (final prediction and bet automation)
"""

import numpy as np
import pandas as pd
# settings to display all columns
pd.set_option("display.max_columns", None)
from sklearn.preprocessing import MinMaxScaler
import itertools

race_details = pd.read_csv('../data/race_details.csv')
dog_results = pd.read_csv('../data/dog_results.csv')

# Clean up the race dataset
race_details = race_details.rename(columns = {'@id': 'FastTrack_RaceId'})
race_details['Distance'] = race_details['Distance'].apply(lambda x: int(x.replace("m", "")))
race_details['date_dt'] = pd.to_datetime(race_details['date'], format = '%d %b %y')
# Clean up the dogs results dataset
dog_results = dog_results.rename(columns = {'@id': 'FastTrack_DogId', 'RaceId': 'FastTrack_RaceId'})

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
box_win_percent.to_csv('../data/box_win_percentage.csv')

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
        temp = temp.sort_index(axis=1)
        rolling_result.loc[pd.IndexSlice[:, pd.Timestamp.now().normalize()], :] = temp.groupby(['FastTrack_DogId','date_dt']).first()

        # Generate list of rolling window feature names (eg: RunTime_norm_min_365D)
        agg_features_cols = [f'{f}_{a}_{rolling_window}' for f, a in itertools.product(features, aggregates)]
        
        # # Add features to dataset
        # dataset[agg_features_cols] = rolling_result

        # More efficient method to add features to dataset
        rolling_result.columns = agg_features_cols
        dataset = pd.concat([dataset,rolling_result],axis = 1)
        
        # Keep track of generated feature names
        feature_cols.extend(agg_features_cols)

# print(feature_cols) 
# feature_cols = ['speed_index', 'box_win_percent', 'RunTime_norm_min_28D', 'RunTime_norm_max_28D', 'RunTime_norm_mean_28D', 'RunTime_norm_median_28D', 'RunTime_norm_std_28D', 'SplitMargin_norm_min_28D', 'SplitMargin_norm_max_28D', 'SplitMargin_norm_mean_28D', 'SplitMargin_norm_median_28D', 'SplitMargin_norm_std_28D', 'Place_inv_min_28D', 'Place_inv_max_28D', 'Place_inv_mean_28D', 'Place_inv_median_28D', 'Place_inv_std_28D', 'Place_log_min_28D', 'Place_log_max_28D', 'Place_log_mean_28D', 'Place_log_median_28D', 'Place_log_std_28D', 'Prizemoney_norm_min_28D', 'Prizemoney_norm_max_28D', 'Prizemoney_norm_mean_28D', 'Prizemoney_norm_median_28D', 'Prizemoney_norm_std_28D', 'RunTime_norm_min_91D', 'RunTime_norm_max_91D', 'RunTime_norm_mean_91D', 'RunTime_norm_median_91D', 'RunTime_norm_std_91D', 'SplitMargin_norm_min_91D', 'SplitMargin_norm_max_91D', 'SplitMargin_norm_mean_91D', 'SplitMargin_norm_median_91D', 'SplitMargin_norm_std_91D', 'Place_inv_min_91D', 'Place_inv_max_91D', 'Place_inv_mean_91D', 'Place_inv_median_91D', 'Place_inv_std_91D', 'Place_log_min_91D', 'Place_log_max_91D', 'Place_log_mean_91D', 'Place_log_median_91D', 'Place_log_std_91D', 'Prizemoney_norm_min_91D', 'Prizemoney_norm_max_91D', 'Prizemoney_norm_mean_91D', 'Prizemoney_norm_median_91D', 'Prizemoney_norm_std_91D', 'RunTime_norm_min_365D', 'RunTime_norm_max_365D', 'RunTime_norm_mean_365D', 'RunTime_norm_median_365D', 'RunTime_norm_std_365D', 'SplitMargin_norm_min_365D', 'SplitMargin_norm_max_365D', 'SplitMargin_norm_mean_365D', 'SplitMargin_norm_median_365D', 'SplitMargin_norm_std_365D', 'Place_inv_min_365D', 'Place_inv_max_365D', 'Place_inv_mean_365D', 'Place_inv_median_365D', 'Place_inv_std_365D', 'Place_log_min_365D', 'Place_log_max_365D', 'Place_log_mean_365D', 'Place_log_median_365D', 'Place_log_std_365D', 'Prizemoney_norm_min_365D', 'Prizemoney_norm_max_365D', 'Prizemoney_norm_mean_365D', 'Prizemoney_norm_median_365D', 'Prizemoney_norm_std_365D']

# Replace missing values with 0
dataset.fillna(0, inplace=True)
print(dataset.head(8))

# Only keep data after 2018-12-01
model_df = dataset.reset_index()
feature_cols = np.unique(feature_cols).tolist()
model_df = model_df[model_df['date_dt'] >= '2018-12-01']

# This line was originally part of Bruno's tutorial, but we don't run it in this script
# model_df = model_df[['date_dt', 'FastTrack_RaceId', 'DogName', 'win', 'StartPrice_probability'] + feature_cols]

# Only train model off of races where each dog has a value for each feature
races_exclude = model_df[model_df.isnull().any(axis = 1)]['FastTrack_RaceId'].drop_duplicates()
model_df = model_df[~model_df['FastTrack_RaceId'].isin(races_exclude)]

# Select todays data and prepare data for easy reference in flumine
todays_data = model_df[model_df['date_dt'] == pd.Timestamp.now().strftime('%Y-%m-%d')]
todays_data['DogName_bf'] = todays_data['DogName'].apply(lambda x: x.replace("'", "").replace(".", "").replace("Res", "").strip())
todays_data.replace({'Sandown (SAP)': 'Sandown Park'}, regex=True, inplace=True)
todays_data = todays_data.set_index(['DogName_bf','Track','RaceNum'])
print(todays_data)

todays_data.to_csv('../data/todays_data.csv', index = True)
