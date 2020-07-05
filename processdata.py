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

def load(spy_path, dia_path, qqq_path):
    spy_df = pd.read_csv(args.spy).drop(columns=['Open','High','Low','Close','Volume'])
    dia_df = pd.read_csv(args.dia).drop(columns=['Open','High','Low','Close','Volume'])
    qqq_df = pd.read_csv(args.qqq).drop(columns=['Open','High','Low','Close','Volume'])
    return spy_df, dia_df, qqq_df

def compute_return(df):
    df['Return'] = np.log(df['Adj Close']) - np.log(df['Adj Close'].shift(periods=1))
    return df

def train_val_test_split(df):
    # set date column as index
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    
    # keeping only the correct date 03/01/2011 to 13/04/2015
    Total_df = df.loc[(cfg.TRAIN_START_DATE <= df.index) & (df.index <= cfg.TEST_STOP_DATE)]
    Training_df = df.loc[(cfg.TRAIN_START_DATE <= df.index) & (df.index <= cfg.TRAIN_STOP_DATE)]
    Test_df = df.loc[(cfg.VAL_START_DATE <= df.index) & (df.index <= cfg.VAL_STOP_DATE)]
    Out_of_sample_df = df.loc[(cfg.TEST_START_DATE <= df.index) & (df.index <= cfg.TEST_STOP_DATE)]
    return Total_df, Training_df, Test_df, Out_of_sample_df

def format_datasets(spydf, diadf, qqqdf):
    for n in ["MLP", "RNN", "PSN"]:
        
        tmp_spydf = spydf.copy()
        tmp_diadf = diadf.copy()
        tmp_qqqdf = qqqdf.copy()
        
        for i in cfg.SPYfeatures[n]:
            tmp_spydf['Return_'+str(i)] = tmp_spydf.Return.shift(i)
        for i in cfg.DIAfeatures[n]:
            tmp_diadf['Return_'+str(i)] = tmp_diadf.Return.shift(i)
        for i in cfg.QQQfeatures[n]:
            tmp_qqqdf['Return_'+str(i)] = tmp_qqqdf.Return.shift(i)
        
        tmp_spydf['Target'] = tmp_spydf.Return
        tmp_diadf['Target'] = tmp_diadf.Return
        tmp_qqqdf['Target'] = tmp_qqqdf.Return
        
        SPY_Total_df, SPY_Training_df, SPY_Test_df, SPY_Out_of_sample_df = train_val_test_split(tmp_spydf)
        DIA_Total_df, DIA_Training_df, DIA_Test_df, DIA_Out_of_sample_df = train_val_test_split(tmp_diadf)
        QQQ_Total_df, QQQ_Training_df, QQQ_Test_df, QQQ_Out_of_sample_df = train_val_test_split(tmp_qqqdf)

        
        os.makedirs(os.path.join("data", "SPY", n), exist_ok=True)
        os.makedirs(os.path.join("data", "DIA", n), exist_ok=True)
        os.makedirs(os.path.join("data", "QQQ", n), exist_ok=True)
        
        utils.save_file(SPY_Training_df, os.path.join("data", "SPY", n, "Train.pkl"))
        utils.save_file(SPY_Test_df, os.path.join("data", "SPY", n, "Valid.pkl"))
        utils.save_file(SPY_Out_of_sample_df, os.path.join("data", "SPY", n, "Test.pkl"))
        
        utils.save_file(DIA_Training_df, os.path.join("data", "DIA", n, "Train.pkl"))
        utils.save_file(DIA_Test_df, os.path.join("data", "DIA", n, "Valid.pkl"))
        utils.save_file(DIA_Out_of_sample_df, os.path.join("data", "DIA", n, "Test.pkl"))
        
        utils.save_file(QQQ_Training_df, os.path.join("data", "QQQ", n, "Train.pkl"))
        utils.save_file(QQQ_Test_df, os.path.join("data", "QQQ", n, "Valid.pkl"))
        utils.save_file(QQQ_Out_of_sample_df, os.path.join("data", "QQQ", n, "Test.pkl"))

if __name__ == "__main__":
    
    spydf, diadf, qqqdf = load(args.spy, args.dia, args.qqq)
    
    spydf = compute_return(spydf)
    diadf = compute_return(diadf)
    qqqdf = compute_return(qqqdf)
    
    # Create and Save Datasets
    format_datasets(spydf, diadf, qqqdf)