
print("Importing libraries")
import sys
import numpy as np
import pandas as pd
import random
import time

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from math import sqrt
import xgboost as xgb

# For data visualizatoin
from bokeh.io import output_notebook, show
from bokeh.models import Title
from bokeh.plotting import figure, output_file, show

import seaborn as sns
# %matplotlib inline

from datetime import datetime, timedelta, date

print("Done")


def import_data():
    print("Reading aggregate consumption data...")
    df=pd.read_csv('~/Documents/work/Active-Learning-TUD-Thesis/mod_datasets/aggregate_consumption.csv', sep=',', header=0, index_col=0, parse_dates=['GMT'], low_memory=False)
    df = df.drop_duplicates()
    print("Done")
    print("Reading weather data...")
    df_midas=pd.read_csv('~/Documents/work/Active-Learning-TUD-Thesis/mod_datasets/midas_weather.csv', sep=',', header=0, index_col=0, parse_dates=['GMT'], low_memory=False)
    df_midas_rs = df_midas.resample('30T').mean()
    df_interpolated = df_midas_rs.interpolate(method='linear')
    df_weather = df_interpolated.loc['2013-01':'2013-12',:]
    df_final = pd.concat([df,df_weather], axis=1)
    print("Done")
    print("Reading LCL consumption data...")
    df_n=pd.read_csv('~/Documents/work/Active-Learning-TUD-Thesis/UKDA-7857-csv/csv/data_collection/data_tables/consumption_n.csv', sep=',', header=0, index_col=0, parse_dates=['GMT'], low_memory=False)
    df_n = df_n.drop_duplicates()
    df_weath = df_interpolated.copy()
    print("Done")
    return df_final