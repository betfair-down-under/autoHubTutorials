"""
    Generate historical prediction for our simulator in how to automate 5
    Uses the model we trained in the Greyhound modelling tutorial and historical data from part_2
    Saves historical predictions as a csv file
"""

import pandas as pd
from joblib import load

model_df = pd.read_csv('../data/model_df.csv')
feature_cols = ['speed_index', 'box_win_percent', 'RunTime_norm_min_28D', 'RunTime_norm_max_28D', 'RunTime_norm_mean_28D', 'RunTime_norm_median_28D', 'RunTime_norm_std_28D', 'SplitMargin_norm_min_28D', 'SplitMargin_norm_max_28D', 'SplitMargin_norm_mean_28D', 'SplitMargin_norm_median_28D', 'SplitMargin_norm_std_28D', 'Place_inv_min_28D', 'Place_inv_max_28D', 'Place_inv_mean_28D', 'Place_inv_median_28D', 'Place_inv_std_28D', 'Place_log_min_28D', 'Place_log_max_28D', 'Place_log_mean_28D', 'Place_log_median_28D', 'Place_log_std_28D', 'Prizemoney_norm_min_28D', 'Prizemoney_norm_max_28D', 'Prizemoney_norm_mean_28D', 'Prizemoney_norm_median_28D', 'Prizemoney_norm_std_28D', 'RunTime_norm_min_91D', 'RunTime_norm_max_91D', 'RunTime_norm_mean_91D', 'RunTime_norm_median_91D', 'RunTime_norm_std_91D', 'SplitMargin_norm_min_91D', 'SplitMargin_norm_max_91D', 'SplitMargin_norm_mean_91D', 'SplitMargin_norm_median_91D', 'SplitMargin_norm_std_91D', 'Place_inv_min_91D', 'Place_inv_max_91D', 'Place_inv_mean_91D', 'Place_inv_median_91D', 'Place_inv_std_91D', 'Place_log_min_91D', 'Place_log_max_91D', 'Place_log_mean_91D', 'Place_log_median_91D', 'Place_log_std_91D', 'Prizemoney_norm_min_91D', 'Prizemoney_norm_max_91D', 'Prizemoney_norm_mean_91D', 'Prizemoney_norm_median_91D', 'Prizemoney_norm_std_91D', 'RunTime_norm_min_365D', 'RunTime_norm_max_365D', 'RunTime_norm_mean_365D', 'RunTime_norm_median_365D', 'RunTime_norm_std_365D', 'SplitMargin_norm_min_365D', 'SplitMargin_norm_max_365D', 'SplitMargin_norm_mean_365D', 'SplitMargin_norm_median_365D', 'SplitMargin_norm_std_365D', 'Place_inv_min_365D', 'Place_inv_max_365D', 'Place_inv_mean_365D', 'Place_inv_median_365D', 'Place_inv_std_365D', 'Place_log_min_365D', 'Place_log_max_365D', 'Place_log_mean_365D', 'Place_log_median_365D', 'Place_log_std_365D', 'Prizemoney_norm_min_365D', 'Prizemoney_norm_max_365D', 'Prizemoney_norm_mean_365D', 'Prizemoney_norm_median_365D', 'Prizemoney_norm_std_365D']
brunos_model = load('../data/logistic_regression.joblib')

# Generate predictions like normal
# Range of dates that we want to simulate later '2022-03-01' to '2022-04-01'
model_df['date_dt'] = pd.to_datetime(model_df['date_dt'])
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
backtest.to_csv('../data/backtest.csv', index=False) # Csv format
