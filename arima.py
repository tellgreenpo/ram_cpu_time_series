import pandas as pd
import statsmodels.api as sm
from itertools import product
from tqdm.notebook import tqdm




# Train many SARIMA models to find the best set of parameters
def optimize_SARIMA(df,targetFeature,parameters_list, d, D, s):

    results = []
    best_aic = float('inf')

    for param in tqdm(parameters_list):
        try: model = sm.tsa.statespace.SARIMAX(df[targetFeature], order=(param[0], d, param[1]),
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




def compute(df,targetFeature):
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

    result_table = optimize_SARIMA(df,targetFeature,parameters_list, d, D, s)
    p, q, P, Q = result_table.parameters[0]

    # Fit the model with the best parameters
    best_model = sm.tsa.statespace.SARIMAX(df[targetFeature], order=(p, d, q),
                                        seasonal_order=(P, D, Q, s)).fit(disp=-1)

    return best_model
