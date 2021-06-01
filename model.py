# Importing the libraries


   
import numpy as np
from numpy import log
import matplotlib.pyplot as plt
import pandas as pd
import pickle
from datetime import datetime
import plotly.express as px

from statsmodels.tsa.holtwinters import ExponentialSmoothing




dataset = pd.read_csv('data3.csv')




dataset['x_time']=pd.to_datetime(dataset['x_time'],format="%Y-%m-%dT%H")


dataset['x_time'] = dataset['x_time'].dt.strftime("%Y-%m-%d %H:%M")

dataset['x_time']=pd.to_datetime(dataset['x_time'])





datacopy= pd.DataFrame(columns=['x_time', 'apiTotalTime'])

dataset=dataset.groupby(['x_time'],as_index=False).sum()
dataset.reset_index()


from statsmodels.tsa.stattools import adfuller
def check_stationarity(timeseries):
    result = adfuller(timeseries,autolag='AIC')
    print('The test statistic: %f' % result[0])
    print('p-value: %f' % result[1])
    print('Critical Values:')
    for key, value in result[4].items():
        print('%s: %.3f' % (key, value))
check_stationarity(dataset['apiTotalTime'])




#wtrain our modeldata.

from statsmodels.tsa.arima_model import ARIMA
model = ARIMA(dataset['apiTotalTime'], order=(1, 1,1))
model_fit = model.fit()




#forecast = model_fit.forecast(steps=7)




#Fitting model with trainig data

# Saving model to disk
pickle.dump(model_fit, open('modell.pkl','wb'))






#--------------------------------------#--------------------


#------------hwes-----------

model_hwes = ExponentialSmoothing(dataset['apiTotalTime'])
# fit the model
model_hwes_fit=model_hwes.fit()


# Saving model to disk
pickle.dump(model_hwes_fit, open('model_hwes.pkl','wb'))



  


