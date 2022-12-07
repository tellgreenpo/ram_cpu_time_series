import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

from sklearn.metrics import r2_score, median_absolute_error, mean_absolute_error
from sklearn.metrics import median_absolute_error, mean_squared_error, mean_squared_log_error

from scipy.optimize import minimize
import statsmodels.tsa.api as smt
import statsmodels.api as sm


from tqdm.notebook import tqdm

from itertools import product

import arima


nodes = ['A','B','C']

# 3 seconds lag

def clean(file):
    df = pd.read_csv(file,sep=';')
    toDrop = [name for name in df.columns if ('RAM' not in name) and ('CPU' not in name)]
    print(toDrop)
    return df.drop(labels=toDrop,axis=1)

def write_csv(df):
    df.to_csv('./RAM_CPU_value.csv')

def plot(df,node):
    ram = []
    cpu = []
    if node in nodes:
        ram = df['RAM_{0}'.format(node)]
        cpu = df['CPU_{0}'.format(node)]
    plt.plot(ram[:100])
    plt.show()


def dickey_fuller_test(series):
    fuller = adfuller(series,autolag='AIC')
    return fuller[1]

def print_p_values(df,nodes):
    for node in nodes:
        pValueRam = dickey_fuller_test(df['RAM_{0}'.format(node)].values)
        pValueCpu = dickey_fuller_test(df['CPU_{0}'.format(node)].values)
        print('RAM_{0} p value: {1}'.format(node,str(pValueRam)))
        print('CPU_{0} p value: {1}'.format(node,str(pValueCpu)))


def time_series_plot(y, lags=None, figsize=(12, 7), syle='bmh'):

    if not isinstance(y, pd.Series):
        y = pd.Series(y)

    with plt.style.context(style='bmh'):
        fig = plt.figure(figsize=figsize)
        layout = (2,2)
        ts_ax = plt.subplot2grid(layout, (0,0), colspan=2)
        acf_ax = plt.subplot2grid(layout, (1,0))
        pacf_ax = plt.subplot2grid(layout, (1,1))

        y.plot(ax=ts_ax)
        p_value = sm.tsa.stattools.adfuller(y)[1]
        ts_ax.set_title('Time Series Analysis Plots\n Dickey-Fuller: p={0:.5f}'.format(p_value))
        smt.graphics.plot_acf(y, lags=lags, ax=acf_ax)
        smt.graphics.plot_pacf(y, lags=lags, ax=pacf_ax)
        plt.tight_layout()
        plt.show()


# to Test
def sample(df,reductionStep):
    toReturn = pd.DataFrame(columns=list(df.columns))
    print(pd.DataFrame(df.iloc[2,]))
    # for i in range(0,df.shape[0],reductionStep):
        # toReturn = pd.concat((toReturn,df.iloc[i,]),axis=0)
    return toReturn

# Difference transform for each node
# def difference_transform(df):
#     for node in df.columns:
#         if node != 'Index':
#             data = []
#             df['{0}_Dif'.format(node)] =


def exploration():
    df = pd.read_csv('RAM_CPU_value.csv',sep=',')
    df.rename(columns= {df.columns[0] : "Index"},inplace=True)
    sampleDf = df.loc[df.Index % 1 == 0]
    # plot(df,'C')

    time_series_plot(sampleDf.CPU_A[:200],lags=50)

# sampledDf = sample(df,4)
# print(sampleDf.head())

# time_series_plot(sampledDf.RAM_A,lags = 100)

# train_model(df)



# TODO - SVM to test
