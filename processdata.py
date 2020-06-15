import os
import pickle
import numpy as np
import pandas as pd
from functools import reduce
import config as cfg

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-spy", type=str, help="Path to the raw SPY data file", required=True)
parser.add_argument("-dia", type=str, help="Path to the raw DIA data file", required=True)
parser.add_argument("-qqq", type=str, help="Path to the raw QQQ data file", required=True)
args = parser.parse_args()

def load_and_merge(spy_path, dia_path, qqq_path):
    spy_df = pd.read_csv(args.spy).drop(columns=['Open','High','Low','Adj Close','Volume'])
    dia_df = pd.read_csv(args.dia).drop(columns=['Open','High','Low','Adj Close','Volume'])
    qqq_df = pd.read_csv(args.qqq).drop(columns=['Open','High','Low','Adj Close','Volume'])
    
    spy_df.columns=spy_df.columns.map(lambda x : x+'_spy' if x !='Date' else x)
    dia_df.columns=dia_df.columns.map(lambda x : x+'_dia' if x !='Date' else x)
    qqq_df.columns=qqq_df.columns.map(lambda x : x+'_qqq' if x !='Date' else x)
    
    dfs = [dia_df, qqq_df, spy_df]
    df = reduce(lambda left,right: pd.merge(left,right,on='Date'), dfs)
    
    return df

def compute_return(df):
    df['Return'] = np.log(df['Adj Close']) - np.log(df['Adj Close'].shift(periods=1))
    pass

def compute_stats(df):
    pass

def train_val_test_split(df):
    # set date column as index
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    
    # keeping only the correct date 03/01/2011 to 13/04/2015
    Total_df = df.loc[('2011-01-03' <= df.index) & (df.index <= '2015-04-13')]
    Training_df = df.loc[('2011-01-03' <= df.index) & (df.index <= '2012-12-31')]
    Test_df = df.loc[('2013-01-02' <= df.index) & (df.index <= '2013-12-31')]
    Out_of_sample_df = df.loc[('2014-01-02' <= df.index) & (df.index <= '2015-04-13')]
    pass

def save_dataset(ds, filename):
    with open(filename+'.pkl', 'wb') as f:
        pickle.dump(ds, f, pickle.HIGHEST_PROTOCOL)
    print("{} saved".format(filename))

if __name__ == "__main__":
    
    df = load_and_merge(args.spy, args.dia, args.qqq)
    print(df)
    
    
    
    
    