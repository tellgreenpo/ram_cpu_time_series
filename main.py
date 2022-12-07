import arima
import exploratory
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, median_absolute_error, mean_absolute_error, mean_absolute_percentage_error
from sklearn.metrics import median_absolute_error, mean_squared_error, mean_squared_log_error

def test():
    df = pd.read_csv('RAM_CPU_value.csv',sep=',')
    df.rename(columns= {df.columns[0] : "Index"},inplace=True)
    # Test for CPU_A
    shortDf = df.iloc[:100,]
    model = arima.compute(shortDf,'CPU_A')
    print(model.summary())

def test2():
    df = pd.read_csv('RAM_CPU_value.csv',sep=',')
    df.rename(columns= {df.columns[0] : "Index"},inplace=True)
    trueValue,prediction = arima.evaluate_error(df,(2,1,3),(0,1,[1,2,3,4],5))
    trueValue = trueValue.tolist()
    prediction = prediction.tolist()
    trueValue.pop(-1)
    prediction.pop(0)
    resultsDf= pd.DataFrame()
    resultsDf['True'] = trueValue
    resultsDf['Prediction'] = prediction
    plt.plot(trueValue,label = "true value")
    plt.plot(prediction,label = "prediction")
    plt.show()
    print(resultsDf.head(10))
    mape = mean_absolute_percentage_error(trueValue,prediction)
    print(mape)
test2()
