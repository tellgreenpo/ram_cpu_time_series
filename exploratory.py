import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller

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
        print('RAM_{0} p value: {1}'.format(node,pValueRam))
        print('CPU_{0} p value: {1}'.format(node,pValueCpu))



df = clean('./value.csv')
print_p_values(df,nodes)
