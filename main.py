import arima
import exploratory
import pandas as pd

def test():
    df = pd.read_csv('RAM_CPU_value.csv',sep=',')
    df.rename(columns= {df.columns[0] : "Index"},inplace=True)
    # Test for CPU_A
    shortDf = df['CPU_A'][:200]
    model = arima.compute(shortDf,'CPU_A')
    print(model.summary())

test()
