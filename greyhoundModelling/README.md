# Betfair Greyhound Modelling

## Project Setup

This project assume you have a developmemnt environment with [Python](https://www.python.org/downloads/) and [pip](https://pip.pypa.io/en/stable/installation/) installed

Clone project
```
git clone https://github.com/betfair-down-under/autoHubTutorials.git
```

Install python libraries
```
cd greyhound-modelling
pip install --user -r requirements.txt
```

Create a configuration file named `.env` in the root folder and set required values:
```
FAST_TRACK_API_KEY=<your key>
```

Launch notebook
```
jupyter notebook
```

## Notebooks

Below is the list of notebooks with detailed examples

### Logistic Regression

[logistic_regression](notebook/logistic_regression.ipynb) provides a step-by-step tutorial from fetching Greyhound racing data from FastTrack API to generating win probabilities using [Scikit-learn](https://scikit-learn.org/stable/) and various Regression techniques.

### Feature Importance

Evaluates feature importance for each of the regression models

## Complete Code

Both Logistic Regression and Feature Importance notebooks are available as a `.py` file, named `main.py` available in the root directory.

Run python script

`py main.py`
