# -*- coding: utf-8 -*-
import warnings
import itertools
import numpy as np
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
plt.style.use('fivethirtyeight')
import pandas as pd
import statsmodels.api as sm
import matplotlib


matplotlib.rcParams['xtick.labelsize'] = 10
matplotlib.rcParams['ytick.labelsize'] = 10
df_all = pd.read_csv("state_wise_daily.csv", parse_dates = ['Date'])
#df_air = pd.read_csv("C:\\Users\\SKD\\COVID19Monitor\\airquality.csv")
df = df_all[['Date','Status','TT']]
df['Date'] = df['Date'].dt.date
plt.plot(df['TT']) # Initial Visualisation
df.info()
# Range of our data
print('Data from: {} to {}'.format(df['Date'].min(), df['Date'].max()))

# check null values
df.isnull().sum()

# df = df.drop(df[(df['Status'] == 'Recovered') & (df['Status'] == 'Deceased')].index, inplace = True)
df = df.drop(df[(df.Status == 'Recovered') | (df.Status == 'Deceased')].index)
df = df.reset_index(drop = True)

plt.plot(df['Date'], df['TT']) # Only Confirmed cases

#aggregate cases by date
df = df.groupby('Date')['TT'].sum().reset_index()
df = df.set_index('Date')
df.index = pd.to_datetime(df.index)
df.plot(figsize = (8,5))
plt.show()

df_decom = sm.tsa.seasonal_decompose(df, model='additive')
fig = df_decom.plot()
fig.set_size_inches(10,12)
plt.show()

#check stationarity
from statsmodels.tsa.stattools import adfuller

def test_stationarity(timeseries):  
    #Determing rolling statistics
    rolmean = timeseries.rolling(window=7).mean()
    rolstd = timeseries.rolling(window=7).std()

    #Plot rolling statistics:
    orig = plt.plot(timeseries, color='blue',label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show(block=False)
    
    #Perform Dickey-Fuller test:
    #Here the null hypothesis is that the TS is non-stationary.
    #If the ‘Test Statistic’ is less than the ‘Critical Value’, we can 
    #reject the null hypothesis and say that the series is stationary (compare signed values not absolute).
    print('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries.iloc[:, 0].values, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print(dfoutput)

#Though the variation in standard deviation is small, mean is clearly increasing 
#with time and this is not a stationary series.
test_stationarity(df)

#Eliminate the trend

#Take log transform
df_log = np.log(df)
df_sqrt = df**0.5
plt.plot(df_log)

#Moving average - Smoothing
moving_avg = df_log.rolling(7).mean()
plt.plot(df_log)
plt.plot(moving_avg, color='red')

df_log_moving_avg_diff = df_log - moving_avg
df_log_moving_avg_diff.head(7)

df_log_moving_avg_diff.dropna(inplace=True)
test_stationarity(df_log_moving_avg_diff)
# Now the TS is stationary


# A drawback can be the forecast would depend on past 7 days
# We will use another moving average where more recent values are given a higher weightage
# Exponentially weighted moving average
expwighted_avg = df_sqrt.ewm(halflife=7).mean()
plt.plot(df_sqrt)
plt.plot(expwighted_avg, color='red')

df_log_ewma_diff = df_sqrt - expwighted_avg
test_stationarity(df_log_ewma_diff)

#ARIMA

#determine p, d, q
from statsmodels.tsa.stattools import acf, pacf
lag_acf = acf(df_log_ewma_diff, nlags=20)
lag_pacf = pacf(df_log_ewma_diff, nlags=20, method='ols')

#Plot ACF: 
f = plt.figure(figsize=(10,3))

p1 = f.add_subplot(121) 
p1.plot(lag_acf)
p1.axhline(y=0,linestyle='--',color='gray')
p1.axhline(y=-1.96/np.sqrt(len(df_log_ewma_diff)),linestyle='--',color='gray')
p1.axhline(y=1.96/np.sqrt(len(df_log_ewma_diff)),linestyle='--',color='gray')
p1.title.set_text('Autocorrelation Function')

#Plot PACF:

p2 = f.add_subplot(122)
p2.plot(lag_pacf)
p2.axhline(y=0,linestyle='--',color='gray')
p2.axhline(y=-1.96/np.sqrt(len(df_log_ewma_diff)),linestyle='--',color='gray')
p2.axhline(y=1.96/np.sqrt(len(df_log_ewma_diff)),linestyle='--',color='gray')
p2.title.set_text('Partial Autocorrelation Function')
f.tight_layout()

#p - The lag value where the PACF chart crosses the upper confidence interval for the first time.
#q - The lag value where the ACF chart crosses the upper confidence interval for the first time.
#from the plots we can see that p is 2 and q is 4
from statsmodels.tsa.arima_model import ARIMA
model = ARIMA(df_log, order=(2, 1, 0))  
results_ARIMA = model.fit(disp=-1)  
plt.plot(df_log_ewma_diff)
plt.plot(results_ARIMA.fittedvalues, color='red')
plt.title('RSS: %.4f'% sum((results_ARIMA.fittedvalues-df_log_ewma_diff)**2))

predictions_ARIMA_diff = pd.Series(results_ARIMA.fittedvalues, copy=True)
print(predictions_ARIMA_diff.head())

predictions_ARIMA_diff_cumsum = predictions_ARIMA_diff.cumsum()
print (predictions_ARIMA_diff_cumsum.head())

predictions_ARIMA_log = pd.Series(df_log.ix[0], index=df_log.index)
predictions_ARIMA_log[0] = df_log.ix[0]
predictions_ARIMA_log = predictions_ARIMA_log.add(predictions_ARIMA_diff, fill_value=0)
predictions_ARIMA_log = predictions_ARIMA_log.cumsum()
predictions_ARIMA_log.head()


predictions_ARIMA = np.exp(predictions_ARIMA_log)
plt.plot(df)
plt.plot(predictions_ARIMA)
plt.title('RMSE: %.4f'% np.sqrt(sum((predictions_ARIMA-df)**2)/len(df)))