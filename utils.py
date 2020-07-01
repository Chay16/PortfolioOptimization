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
#     y_true[y_true == 0] = 0.000001 # to avoid dividing by 0
#     y_true = y_true[y_true != 0] # to delete when y_true = 0
    zeros = np.where(y_true==0)
    y_truebis = np.delete(y_true, zeros)
    y_predbis = np.delete(y_pred, zeros)
    
#     return np.mean(np.abs((y_true - y_pred)/y_true))
    return np.mean(np.abs((y_truebis - y_predbis)/y_truebis))

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
def jarque_bera_t_stat(x):
    return (list(jarque_bera(x))[0]) # t stat

def jarque_bera_p_value(x):
    return (list(jarque_bera(x))[1]) # p value

# Augmented Dickey-Fuller, null hypothesis: the serie is stationnary
def adf_t_stat(x):
    return (list(adfuller(x))[0]) # t stat

def adf_p_value(x):
    return (list(adfuller(x))[1]) # p value

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
    
    if (v-w)**0.5 == 0:
        return (50) 

    PT = (pyz - p)/(v-w)**0.5
    return(PT)

# Diebold-Mariano test: null hypothesis equal predictive accuracy between two forecasts
# copied from John Tsang https://github.com/johntwk/Diebold-Mariano-Test/blob/master/dm_test.py
def dm_test(actual_lst, pred1_lst, pred2_lst, h = 1, crit="MSE", power = 2):
    # Routine for checking errors
    def error_check():
        rt = 0
        msg = ""
        # Check if h is an integer
        if (not isinstance(h, int)):
            rt = -1
            msg = "The type of the number of steps ahead (h) is not an integer."
            return (rt,msg)
        # Check the range of h
        if (h < 1):
            rt = -1
            msg = "The number of steps ahead (h) is not large enough."
            return (rt,msg)
        len_act = len(actual_lst)
        len_p1  = len(pred1_lst)
        len_p2  = len(pred2_lst)
        # Check if lengths of actual values and predicted values are equal
        if (len_act != len_p1 or len_p1 != len_p2 or len_act != len_p2):
            rt = -1
            msg = "Lengths of actual_lst, pred1_lst and pred2_lst do not match."
            return (rt,msg)
        # Check range of h
        if (h >= len_act):
            rt = -1
            msg = "The number of steps ahead is too large."
            return (rt,msg)
        # Check if criterion supported
        if (crit != "MSE" and crit != "MAPE" and crit != "MAD" and crit != "poly"):
            rt = -1
            msg = "The criterion is not supported."
            return (rt,msg)  
        # Check if every value of the input lists are numerical values
#         from re import compile as re_compile
#         comp = re_compile("^\d+?\.\d+?$")  
#         def compiled_regex(s):
#             """ Returns True is string is a number. """
#             if comp.match(s) is None:
#                 return s.isdigit()
#             return True
#         for actual, pred1, pred2 in zip(actual_lst, pred1_lst, pred2_lst):
#             is_actual_ok = compiled_regex(str(abs(actual)))
#             is_pred1_ok = compiled_regex(str(abs(pred1)))
#             is_pred2_ok = compiled_regex(str(abs(pred2)))
#             if (not (is_actual_ok and is_pred1_ok and is_pred2_ok)):
#                 msg = "An element in the actual_lst, pred1_lst or pred2_lst is not numeric."
#                 rt = -1
#                 return (rt,msg)
        return (rt,msg)
    
    # Error check
    error_code = error_check()
    # Raise error if cannot pass error check
    if (error_code[0] == -1):
        raise SyntaxError(error_code[1])
        return
    # Import libraries
    from scipy.stats import t
    import collections
    import pandas as pd
    import numpy as np
    
    # Initialise lists
    e1_lst = []
    e2_lst = []
    d_lst  = []
    
    # convert every value of the lists into real values
    actual_lst = pd.Series(actual_lst).apply(lambda x: float(x)).tolist()
    pred1_lst = pd.Series(pred1_lst).apply(lambda x: float(x)).tolist()
    pred2_lst = pd.Series(pred2_lst).apply(lambda x: float(x)).tolist()
    
    # Length of lists (as real numbers)
    T = float(len(actual_lst))
    
    # construct d according to crit
    if (crit == "MSE"):
        for actual,p1,p2 in zip(actual_lst,pred1_lst,pred2_lst):
            e1_lst.append((actual - p1)**2)
            e2_lst.append((actual - p2)**2)
        for e1, e2 in zip(e1_lst, e2_lst):
            d_lst.append(e1 - e2)
    elif (crit == "MAD"):
        for actual,p1,p2 in zip(actual_lst,pred1_lst,pred2_lst):
            e1_lst.append(abs(actual - p1))
            e2_lst.append(abs(actual - p2))
        for e1, e2 in zip(e1_lst, e2_lst):
            d_lst.append(e1 - e2)
    elif (crit == "MAPE"):
        for actual,p1,p2 in zip(actual_lst,pred1_lst,pred2_lst):
            e1_lst.append(abs((actual - p1)/actual))
            e2_lst.append(abs((actual - p2)/actual))
        for e1, e2 in zip(e1_lst, e2_lst):
            d_lst.append(e1 - e2)
    elif (crit == "poly"):
        for actual,p1,p2 in zip(actual_lst,pred1_lst,pred2_lst):
            e1_lst.append(((actual - p1))**(power))
            e2_lst.append(((actual - p2))**(power))
        for e1, e2 in zip(e1_lst, e2_lst):
            d_lst.append(e1 - e2)    
    
    # Mean of d        
    mean_d = pd.Series(d_lst).mean()
    
    # Find autocovariance and construct DM test statistics
    def autocovariance(Xi, N, k, Xs):
        autoCov = 0
        T = float(N)
        for i in np.arange(0, N-k):
              autoCov += ((Xi[i+k])-Xs)*(Xi[i]-Xs)
        return (1/(T))*autoCov
    gamma = []
    for lag in range(0,h):
        gamma.append(autocovariance(d_lst,len(d_lst),lag,mean_d)) # 0, 1, 2
    V_d = (gamma[0] + 2*sum(gamma[1:]))/T
    DM_stat=V_d**(-0.5)*mean_d
    harvey_adj=((T+1-2*h+h*(h-1)/T)/T)**(0.5)
    DM_stat = harvey_adj*DM_stat
    # Find p-value
    p_value = 2*t.cdf(-abs(DM_stat), df = T - 1)
    
    # Construct named tuple for return
    dm_return = collections.namedtuple('dm_return', 'DM p_value')
    
    rt = dm_return(DM = DM_stat, p_value = p_value)
    
    return rt

# Maximum Drawdown
def MDD(df, column, window, everything=False):
    # Calculate the max drawdown in the past window days for each day in the series.
    # Use min_periods=1 if you want to let the first days data have an expanding window
    Roll_Max = df[column].rolling(window, min_periods=1).max()
    Daily_Drawdown = df[column]/Roll_Max - 1.0

    # Next we calculate the minimum (negative) daily drawdown in that window.
    # Again, use min_periods=1 if you want to allow the expanding window
    Max_Daily_Drawdown = Daily_Drawdown.rolling(window, min_periods=1).min()
    MDD = min(Max_Daily_Drawdown)

    if everything:
        return (Daily_Drawdown, Max_Daily_Drawdown, MDD)
    return (MDD)
