import pandas as pd
import csv
import matplotlib.pyplot as plt

nodes = ['A','B','C']

def clean(file):
    df = pd.read_csv(file,sep=';')
    toDrop = [name for name in df.columns if ('RAM' not in name) and ('CPU' not in name)]
    print(toDrop)
    return df.drop(labels=toDrop,axis=1)

def write_csv(df):
    df.to_csv('./RAM_CPU_value.csv')

df = clean('./value.csv')

def plot(df,node):
    ram = []
    cpu = []
    if node in nodes:
        ram = df['RAM_{0}'.format(node)]
        cpu = df['CPU_{0}'.format(node)]
    plt.plot(ram[:100])
    plt.show() 

plot(df,'A')
print(df['RAM_A'].mean())
print(df['RAM_A'].var())
