import pickle
import numpy as np
import pandas as pd
from torch import Tensor
from torch.utils.data import TensorDataset, DataLoader
from scipy.stats import jarque_bera
from statsmodels.tsa.stattools import adfuller

def save_file(data, filename):
    with open(filename, 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
    print("{} saved".format(filename))

def load_file(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)

def mean_absolute_percentage_error(y_true, y_pred):
    y_true[y_true == 0] = 0.000001 # to avoid dividing by 0
    return np.mean(np.abs((y_true - y_pred)/y_true))

def theilU(y_true, y_pred):
    return np.sqrt(np.mean((y_pred - y_true)**2)) / (np.sqrt(np.mean(y_pred**2)) + np.sqrt(np.mean(y_true**2)))

    
# Convert Dataframe to DataLoader
def DataFrame2DataLoader(df, features_col, target_col, batch_size=4, normalize=False, mu=None, sigma=None):
    tmpdf = df.copy()
    try:
        del tmpdf["Date"]
    except:
        pass
    if normalize:
        tmpdf = (tmpdf - mu)/sigma
    
    target = tmpdf[target_col]
    features = tmpdf[features_col]
    
    dataset = TensorDataset(Tensor(np.array(features)), Tensor(np.array(target)))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    return dataloader

# Jarque-Bera result, null hypothesis: the serie follows a normal distribution
def jarque_bera_p_value(x):
    return (list(jarque_bera(x))[1]) #put 0 to get the t stat

# Augmented Dickey-Fuller, null hypothesis: the serie is stationnary
def adf_p_value(x):
    return (list(adfuller(x))[1]) #put 0 to get the t stat

# Pesaran-Timmermann test: null hypothesis: the model under study has no power on forecasting the relevant ETF return series
def PT_test(y_true, y_pred):
    n = len(y_true)

    dy = y_true.copy()
    dy[dy < 0] = 0
    dy[dy > 0] = 1

    py = np.mean(dy)
    qy = (py*(1-py))/n

    dz = y_pred.copy()
    dz[dz < 0] = 0
    dz[dz > 1] = 1

    pz = np.mean(dz)
    qz = (pz*(1-pz))/n

    p = py*pz + (1-py)*(1-pz)

    v = (p*(1-p))/n
    w = ((2*py-1)**2)*qz + ((2*pz-1)**2)*qy + 4*qy*qz

    dyz = y_true*y_pred.copy()
    dyz[dyz < 0] = 0
    dyz[dyz > 0] = 1
    pyz = np.mean(dyz)

    PT = (pyz - p)/(v-w)**0.5
    return(PT)