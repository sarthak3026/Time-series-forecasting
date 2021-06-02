from numpy.core.defchararray import equal
from numpy.core.numeric import isscalar
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from datetime import datetime
from matplotlib import pylab as plt
import statsmodels.api as sm

df=pd.read_excel("dataForecasting.xlsx")

# label encoding of the categories
df['opened_at']=pd.to_datetime(df['opened_at'],infer_datetime_format=True)
indexedDf_temp=df.set_index('opened_at')
encoder=LabelEncoder()
indexedDf_temp['encoded']=encoder.fit_transform(indexedDf_temp['Category1'].astype(str))
indexedDf=indexedDf_temp.copy()
indexedDf.drop(labels=['Category1'],axis=1,inplace=True)

# stationarity check -- 1.rolmean  2.ADFC
def stationarityCheck(timeseries):
    dftest=sm.tsa.adfuller(timeseries['encoded'], autolag='AIC') 
    dfoutput=pd.Series(dftest[0:4],index=['teststats','pvalue','lags used','no of observations'])
    for key,values in dftest[4].items():
        dfoutput['critical values: %s'%key]=values
    print(dfoutput)
    rolmean=timeseries.rolling(window=365).mean()
    rolsd=timeseries.rolling(window=365).std()
    plotOriginal=plt.plot(timeseries, color='blue',label="original")
    plotMean=plt.plot(rolmean,color="red",label="rolmean")
    plotSd=plt.plot(rolsd,color="black",label="rolsd")
    plt.legend(loc='best')
    plt.show()

# making data stationary
indexedDf_log=np.log(indexedDf)
indexedDf_log.replace([np.inf,-np.inf],np.nan,inplace=True)
indexedDf_log.dropna(inplace=True)
weightedMean=indexedDf_log.ewm(halflife=365,min_periods=0,adjust=True).mean()
logscaleMinusWeightedMean=indexedDf_log-weightedMean
logsScaleShift=indexedDf_log-indexedDf_log.shift()
logsScaleShift.replace([np.inf, -np.inf],np.nan,inplace=True)
logsScaleShift.dropna(inplace=True)
# stationarityCheck(logsScaleShift)

# ARIMA model prediction
decomposition=sm.tsa.seasonal_decompose(indexedDf_log,period=30)
trend=decomposition.trend
seasonal=decomposition.seasonal
residual=decomposition.resid
decomposedResidual=residual
decomposedResidual.dropna(inplace=True)

lag_acf=sm.tsa.acf(logsScaleShift,nlags=40)
lag_pacf=sm.tsa.pacf(logsScaleShift,nlags=40,method="ols")
    # calculation of p value and q value
    # plt.subplot(121)
    # plt.plot(lag_acf)
    # plt.title("acf graph")
    # plt.subplot(122)
    # plt.plot(lag_pacf)
    # plt.title("pacf graph")
    # plt.show()
model=sm.tsa.ARIMA(indexedDf_log,order=(1,1,1))
resultArima=model.fit(disp=-1)
# print('RSS:%s'%sum((resultArima.fittedvalues-logsScaleShift)**2))

predictionARIMA=pd.Series(resultArima.fittedvalues,copy=True)
predictionARIMA_cumsum=predictionARIMA.cumsum()
predictionARIMA_log=pd.Series(indexedDf_log['encoded'],index=indexedDf_log.index)
predictionARIMA_log=predictionARIMA_log.add(predictionARIMA_cumsum,fill_value=0)
prediction_ARIMA=np.exp(predictionARIMA_log)
predictionFinal=resultArima.forecast(steps=1000)

def returnPredictedTickets(predictionFinal):
    encodedArimaPredicted=[]
    for predItems in predictionFinal:
        for flag in predItems:
            if np.isscalar(flag) is False:
                for i in flag:
                    i=np.floor(i)
                    encodedArimaPredicted.append(i)
            else:
                flag=np.floor(flag)
                encodedArimaPredicted.append(flag)

    encodedArimaPredicted=[int(x) for x in encodedArimaPredicted]
    decoder=encoder.inverse_transform(encodedArimaPredicted)
    return decoder

predictedVal=returnPredictedTickets(predictionFinal)
print(predictedVal)

