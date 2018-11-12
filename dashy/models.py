
import warnings                                  # `do not disturbe` mode
warnings.filterwarnings('ignore')

import numpy as np                               # vectors and matrices
import pandas as pd                              # tables and data manipulations
import matplotlib.pyplot as plt                  # plots
import seaborn as sns                            # more plots
import datetime as dt

from dateutil.relativedelta import relativedelta # working with dates with style
from scipy.optimize import minimize              # for function minimization

import statsmodels.formula.api as smf            # statistics and econometrics
import statsmodels.tsa.api as smt
import statsmodels.api as sm
import scipy.stats as scs
# import pandas_profiling

from itertools import product                    # some useful functions
from tqdm import tqdm_notebook
from sklearn.base import BaseEstimator, TransformerMixin
from statsmodels.tsa.api import ExponentialSmoothing
from statsmodels.tsa.seasonal import seasonal_decompose
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from sklearn.metrics import make_scorer


ny = [(1, x) for x in range(1, 9)]
fem = [(3, x) for x in range(8, 12)]
male = [(2, x) for x in range(23, 26)]
jun = [(6, x) for x in range(10, 13)]
jun2 = [(6, x) for x in range(15, 17)] # TAT
nov = [(11, x) for x in range(3, 6)] 

shorter_17 = [(2, 22), (3, 7), (8, 29), (8, 31), (11, 3)]
shorter_18 = [(2, 22), (3, 7), (4, 28), (5, 8), (6, 9), (6, 14), (8, 20), (8, 29)]

# Russian 
rus_17 = [(3, 8), (5, 1), (5, 8), (5, 9), (6, 12)] + ny + male + nov
rus_18 = [(1, 7), (5, 9), (4, 28)] + ny + fem + male + jun + jun2 + nov

# Tatartan 
tat_17 = [(6, 25), (8, 30), (9, 1), (11, 6)]
tat_18 = [(11, 6), (8, 30), (8, 21)] + jun2


rus_17 = set([dt.datetime(2017, m, d) for (m, d) in rus_17])
rus_18 = set([dt.datetime(2018, m, d) for (m, d) in rus_18])
tat_17 = set([dt.datetime(2017, m, d) for (m, d) in tat_17])
tat_18 = set([dt.datetime(2018, m, d) for (m, d) in tat_18])
shorter_17 = set([dt.datetime(2017, m, d) for (m, d) in shorter_17])
shorter_18 = set([dt.datetime(2018, m, d) for (m, d) in shorter_18])

class DatetimeTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, year=True, month=True, day=True, dayofweek=True, 
                 hour=True, minute=True, second=True, russian_holidays=False,
                 weekdays_as_holidays=False, tatar_holidays=False, shortened_day=False, **kwargs):
        self.enc = []
        if year:
            self.enc.append("year")
        if month:
            self.enc.append("month")
        if day:
            self.enc.append("day")
        if dayofweek:
            self.enc.append("dayofweek")
        if hour:
            self.enc.append("hour")
        if minute:
            self.enc.append("minute")
        if second:
            self.enc.append("second")
        
        self.weekdays_as_holidays = weekdays_as_holidays
        self.russian_holidays = russian_holidays
        self.tatar_holidays = tatar_holidays
        self.shortened_day = shortened_day
        
    def _is_tatar_holiday(self, date):
        return date in tat_17 or date in tat_18
    
    def _is_russian_holiday(self, date):
        return date in rus_17 or date in rus_18
    
    def _is_shortened(self, date):
        return date in shorter_17 or date in shorter_18
    
    def fit(self, X, y=None):
        return self

    def transform(self, col):
        result = pd.DataFrame()
        
        scol = pd.Series(col)

        # Parse weekday holidays
        if self.weekdays_as_holidays:
            result["weekday_holiday"] = scol.apply(lambda date: date.weekday() in [5, 6])
            
        # Parse tatarstan holidays
        if self.tatar_holidays:
            result["tatar_holiday"] = scol.apply(lambda date: self._is_tatar_holiday(date))
            
        # Parse russian holidays
        if self.russian_holidays:
            result["russian_holiday"] = scol.apply(lambda date: self._is_russian_holiday(date))
            
        # Parse shortened days
        if self.shortened_day:
            result["shortened"] = scol.apply(lambda date: self._is_shortened(date))
            
        # Parse datetime
        if len(self.enc) != 0:
            for date_type in self.enc:
                result["{}_{}".format("name", date_type)] = getattr(col, date_type)
        
        return result

class ColumnSelector(BaseEstimator, TransformerMixin):
    """
    Transformer to select a single column from the data frame to perform additional transformations on
    Use on text columns in the data
    """
    def __init__(self, key):
        self.key = key

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.key]
    
class IndexSelector(BaseEstimator, TransformerMixin):
    """
    Transformer to select an index only
    """
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.index


import datetime
fitDate = datetime.datetime.now()

upbound_model =  Pipeline([
        ("selector", IndexSelector()),
        ("features", DatetimeTransformer(year=False, month=False, day=True, dayofweek=True,
                                         hour=False, minute=False, second=False, russian_holidays=True,
                                         tatar_holidays=True, shortened_day=True, weekdays_as_holidays=False)),
        ("bounds", GradientBoostingRegressor(loss='quantile', alpha=0.95,
                                n_estimators=500, max_depth=25,
                                learning_rate=.05, min_samples_leaf=10,
                                min_samples_split=5))
])

lowbound_model =  Pipeline([
        ("selector", IndexSelector()),
        ("features", DatetimeTransformer(year=False, month=False, day=True, dayofweek=True,
                                         hour=False, minute=False, second=False, russian_holidays=True,
                                         tatar_holidays=True, shortened_day=True, weekdays_as_holidays=False)),
        ("bounds", GradientBoostingRegressor(loss='quantile', alpha=0.05,
                                n_estimators=500, max_depth=25,
                                learning_rate=.05, min_samples_leaf=10,
                                min_samples_split=5))
])

forecast_model = Pipeline([
        ("selector", IndexSelector()),
        ("features", DatetimeTransformer(year=False, month=False, day=True, dayofweek=True,
                                         hour=False, minute=False, second=False, russian_holidays=True,
                                         tatar_holidays=True, shortened_day=True, weekdays_as_holidays=False)),
        ("forecaster", XGBRegressor())
])

def retrain(client):
    results = list(client.query("SELECT * FROM calls_in"))[0]
    X = pd.DataFrame([(x['time'], x['value']) for x in results])
    X[0] = pd.to_datetime(X[0])
    X.set_index(0, inplace=True)
    X_train, y_train = X, X[1].values

    forecast_model.fit(X_train, y_train)
    upbound_model.fit(X_train, y_train)
    lowbound_model.fit(X_train, y_train)


def predict(X_in):
    dummy_series = pd.DataFrame([(X_in, 0)], columns=["time", "dummy"])
    dummy_series.index = pd.DatetimeIndex(pd.to_datetime(dummy_series['time']))

    pred = forecast_model.predict(dummy_series)[0]
    up_bound = upbound_model.predict(dummy_series)[0]
    lower_bound = lowbound_model.predict(dummy_series)[0]
    return (pred, lower_bound, up_bound)

        