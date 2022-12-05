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


def train_model(df):
    #Set initial values and some bounds
    ps = range(0, 5)
    d = 1
    qs = range(0, 5)
    Ps = range(0, 5)
    D = 1
    Qs = range(0, 5)
    s = 5

    #Create a list with all possible combinations of parameters
    parameters = product(ps, qs, Ps, Qs)
    parameters_list = list(parameters)
    len(parameters_list)

    # Train many SARIMA models to find the best set of parameters
    def optimize_SARIMA(df,parameters_list, d, D, s):
        """
            Return dataframe with parameters and corresponding AIC

            parameters_list - list with (p, q, P, Q) tuples
            d - integration order
            D - seasonal integration order
            s - length of season
        """

        results = []
        best_aic = float('inf')

        for param in tqdm(parameters_list):
            try:
                model = sm.tsa.statespace.SARIMAX(df.CPU_A, order=(param[0], d, param[1]),
                                                seasonal_order=(param[2], D, param[3], s)).fit(disp=-1)
            except:
                continue

            aic = model.aic

            #Save best model, AIC and parameters
            if aic < best_aic:
                best_model = model
                best_aic = aic
                best_param = param
            results.append([param, model.aic])

        result_table = pd.DataFrame(results)
        result_table.columns = ['parameters', 'aic']
        #Sort in ascending order, lower AIC is better
        result_table = result_table.sort_values(by='aic', ascending=True).reset_index(drop=True)

        return result_table

    result_table = optimize_SARIMA(df,parameters_list, d, D, s)

    #Set parameters that give the lowest AIC (Akaike Information Criteria)
    p, q, P, Q = result_table.parameters[0]

    best_model = sm.tsa.statespace.SARIMAX(df.CPU_A, order=(p, d, q),
                                        seasonal_order=(P, D, Q, s)).fit(disp=-1)

    print(best_model.summary())


df = pd.read_csv('RAM_CPU_value.csv',sep=',')
# plot(df,'C')

# time_series_plot(df.RAM_A,lags=100)

train_model(df)
