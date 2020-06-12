import numpy as np
import pandas as pd
from torch import Tensor
from torch.utils.data import TensorDataset, DataLoader

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