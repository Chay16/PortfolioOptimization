import pickle
import numpy as np
import pandas as pd
from torch import Tensor
from torch.utils.data import TensorDataset, DataLoader

def save_file(data, filename):
    with open(filename, 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
    print("{} saved".format(filename))

def load_file(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)

def mean_absolute_percentage_error(y_true, y_pred):
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