import numpy as np
import pandas as pd

def load_data(path, rename=True):
  df = pd.read_csv(path)
  df.latitude = pd.to_numeric(df.latitude, errors='coerse')
  df['date posted'] = pd.to_datetime(df['date posted'], errors='coerce')
  df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
  df['shape'] = df['shape'].astype("category")
  if rename:
    df['duration (seconds)'] = pd.to_numeric(df['duration (seconds)'], errors='coerse', downcast='unsigned')
    df.columns = ['datetime', 'city', 'state', 'country', 'shape', 'duration_sec',
       'duration_long', 'comments', 'date posted', 'latitude',
       'longitude']
  return df

def preprocess_data(df):
  df.comments = fill_na_mode(df.comments)
  df['duration_sec'] = fill_na(df['duration_sec'])
  fill_lr(df.loc[:, ['longitude', 'duration_sec']].values, df.latitude)
  df = df.drop('duration_long', 1)
  df['shape'] = fill_knn(df.loc[:, ['longitude', 'latitude', 'duration_sec']].values, df['shape'].astype(object), k=1)
  return df

def save_df(df, filename="ufo_df.csv"):
  df.to_csv(filename, index=False)