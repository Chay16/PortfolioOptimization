import os
import pickle
import numpy as np
import pandas as pd
from functools import reduce
import config as cfg
import utils

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-spy", type=str, help="Path to the raw SPY data file", required=True)
parser.add_argument("-dia", type=str, help="Path to the raw DIA data file", required=True)
parser.add_argument("-qqq", type=str, help="Path to the raw QQQ data file", required=True)
args = parser.parse_args()

def load_and_merge(spy_path, dia_path, qqq_path):
    spy_df = pd.read_csv(args.spy).drop(columns=['Open','High','Low','Close','Volume'])
    dia_df = pd.read_csv(args.dia).drop(columns=['Open','High','Low','Close','Volume'])
    qqq_df = pd.read_csv(args.qqq).drop(columns=['Open','High','Low','Close','Volume'])
    
    spy_df.columns=spy_df.columns.map(lambda x : x+'_spy' if x !='Date' else x)
    dia_df.columns=dia_df.columns.map(lambda x : x+'_dia' if x !='Date' else x)
    qqq_df.columns=qqq_df.columns.map(lambda x : x+'_qqq' if x !='Date' else x)
    
    dfs = [dia_df, qqq_df, spy_df]
    df = reduce(lambda left,right: pd.merge(left,right,on='Date'), dfs)
    return df
    
def compute_return_bda(df):
    df['Return'] = np.log(df['Adj Close']) - np.log(df['Adj Close'].shift(periods=1))
    for k in range(1,13):
        df['Return' + "_" + str(k)] = df['Return'].shift(periods=k)

def load(spy_path, dia_path, qqq_path):
    spy_df = pd.read_csv(args.spy).drop(columns=['Open','High','Low','Close','Volume'])
    dia_df = pd.read_csv(args.dia).drop(columns=['Open','High','Low','Close','Volume'])
    qqq_df = pd.read_csv(args.qqq).drop(columns=['Open','High','Low','Close','Volume'])
    return spy_df, dia_df, qqq_df

def compute_return_chay(df):
    df['Return'] = np.log(df['Adj Close']) - np.log(df['Adj Close'].shift(periods=1))
    
    """
    for k in range(1,13):
        df['Return' + str(k)] = df['Return'].shift(periods=k)
    
    """ 
    return df

def compute_stats(df):
    Stats_df = pd.DataFrame({'Mean':df['Return'].mean(),
                             'STD':df['Return'].std(),
                             'Skew':df['Return'].skew(),
#                              'Fisher_Kurtosis':kurtosis(df['Return'], fisher=True),
                             'Pearson_Kurtosis':kurtosis(df['Return'], fisher=False),
                             'Jarque-Bera_p_value':jarque_bera_p_value(df['Return']),
                             'ADF_p_value':adf_p_value(df['Return'])})
    return Stats_df

def compute_corr_matrix(df, method_name='pearson'):
     return df.corr(method=method_name)

def train_val_test_split(df):
    # set date column as index
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    
    # keeping only the correct date 03/01/2011 to 13/04/2015
    Total_df = df.loc[(cfg.TRAIN_START_DATE <= df.index) & (df.index <= cfg.TEST_STOP_DATE)]
    Training_df = df.loc[(cfg.TRAIN_START_DATE <= df.index) & (df.index <= cfg.TRAIN_STOP_DATE)]
    Test_df = df.loc[(cfg.VAL_START_DATE <= df.index) & (df.index <= cfg.VAL_START_DATE)]
    Out_of_sample_df = df.loc[(cfg.TEST_START_DATE <= df.index) & (df.index <= cfg.VAL_START_DATE)]
    return Total_df, Training_df, Test_df, Out_of_sample_df

def save_dataset(ds, filename):
    with open(filename+'.pkl', 'wb') as f:
        pickle.dump(ds, f, pickle.HIGHEST_PROTOCOL)
    print("{} saved".format(filename))

if __name__ == "__main__":
    
    spydf, diadf, qqqdf = load(args.spy, args.dia, args.qqq)
    
    spydf = compute_return(spydf)
    diadf = compute_return(diadf)
    qqqdf = compute_return(qqqdf)
    
    networks = ["MLP", "RNN", "PSN"]
    
    for n in networks:
        
    
    
    
    
    
    