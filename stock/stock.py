
import matplotlib
import os
from builtins import reversed
from matplotlib.figure import Figure
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import keras.backend as K
import keras
from sklearn import preprocessing
import datetime
from math import floor
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers import Reshape
from keras.layers import Dense, Embedding
from keras.layers.recurrent import LSTM
from keras.models import model_from_json
from keras import optimizers
from sklearn.preprocessing import KBinsDiscretizer
from keras.datasets import imdb
from keras.layers import GRU, LSTM, CuDNNGRU, CuDNNLSTM, Activation
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.statespace.sarimax import SARIMAX
import matplotlib.dates as mdates
from dateutil.parser import parse

def read_data(file_name):
    dataset = pd.read_csv(file_name)
    return dataset

def data_cleaning_scaling(stock_data):
    stock_data['Date'] = pd.to_datetime(stock_data['Date'])
    stock_data['Close'] = stock_data['Close'].fillna(value=stock_data['Close'].shift(1))
    stock_data['Open'] = stock_data['Open'].fillna(value=stock_data['Open'].shift(1))

    close = stock_data[['Close']].values
    open = stock_data[['Open']].values
    high = stock_data[['HIGH_PRICE']].values
    low = stock_data[['LOW_PRICE']].values
    scaleropen = preprocessing.MinMaxScaler()
    scalerclose = preprocessing.MinMaxScaler()
    stock_data_scaled_close = scalerclose.fit_transform(close)
    stock_data_scaled1_open = scaleropen.fit_transform(open)
    return stock_data_scaled_close, stock_data_scaled1_open,high,low,open,close,scaleropen,scalerclose


def data_splitting(scaled_data_close, scaled_data_open):
    x_train = []
    y_train = []

    for i in range(0, scaled_data_close.shape[0] - 120):
        x_train.append(scaled_data_close[i:(i+120)])

        y_train.append(scaled_data_open[i + 120])

    x_train, y_train = np.array(x_train), np.array(y_train)


    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2, shuffle=False)
    return x_train, x_test, y_train, y_test




def GRU_model(x_train, y_train):
    print('Build model...')
    model = Sequential()
    model.add(LSTM(units=64, activation='relu',input_shape=(x_train.shape[1],1)))
    model.add(Dense(units=1, activation="linear"))
    model.compile(optimizer="RMSProp", loss="mean_squared_error")
    model.fit(x_train, y_train, batch_size=8, epochs=50,  verbose=1)
    # serialize model to JSON
    model_json = model.to_json()
    with open("gruclassficationmodel1.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("gruclassficationmodel1.h5")
    print("Saved model to disk")


def GRU_model_load(testingX, testingY,trainingX):

    json_file = open('/home/reem/stock_site/stock/gruclassficationmodel1.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("/home/reem/stock_site/stock/gruclassficationmodel1.h5")
    print("Loaded model from disk")

    score = 0
    pred = loaded_model.predict(testingX)
    for i in range(0, len(pred)):
        inner = abs(testingY[i] - pred[i])
        inner = inner / testingY[i]
        if (inner <= .05):
            score += 1
    print(score, "before", len(pred), "\n the accuracy", score / len(pred))
    return pred


def GRU_predict(dataclose):
    json_file = open('/home/reem/stock_site/stock/gruclassficationmodel1.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    i=1
    test=[]
    test1=[]
    loaded_model.load_weights("/home/reem/stock_site/stock/gruclassficationmodel1.h5")

    while(i<=120):

        dataclose[-i]=dataclose[-i].tolist()

        test.append(dataclose[-i])
        i=i+1

    for a in reversed(test):
        test1.append(a)

    test1=np.array(test1)
    test1 = np.reshape(test1, (1, test1.shape[0], test1.shape[1]))
    list_of_predit=[]
    print(test1.shape)
    for i in range (3):

        pred = loaded_model.predict(test1)
        list_of_predit.append(pred)
        test1=np.delete(test1,0)
        test1=np.append(test1,pred)
        test1 = np.reshape(test1, (1, test1.shape[0], 1))
        print(test1.shape)

    print("list_of_predit",list_of_predit)

    #print("the value",pred)
    return list_of_predit ,test1

def data_plotting(dates, pred, testingY,dates1,list_3days):

    fig, ax = plt.subplots()

    ax.plot(dates[-11:-1], testingY[-11:-1], label='real price')
    print(dates[-11:-1])
    ax.plot(dates1, list_3days, color='r', label='prediction')
    plt.gcf().autofmt_xdate()


    '''
    fig = Figure()
    ax = fig.add_subplot(1, 1, 1)

    ax.plot(dates, pred, 'red')
    ax.plot(dates, testingY, 'blue')
    ax.plot(dates1,list_3days,'red')
    ax.set_xlabel('Date')
    ax.set_ylabel('open Price')
    '''
    '''
    plt.figure('fig1')
    plt.plot(dates, pred, 'red')
    plt.plot(dates, testingY, 'blue')
    plt.xlabel('Date')
    plt.ylabel('open price')
  #  plt.show()
  '''
    return fig , ax



def classification (high, low ,openprice,closeprice,prediction):
    coun = 0

    list1 = []
    open1 = []
    condition=[]
    predictionprice=[]
    eq=[]
    for i in range(len(openprice) -1):
        x = (((high[i] + low[i]) / 2) - openprice[i])
        y=(((openprice[i] + closeprice[i]) / 2) - openprice[i])
        if x >= 0:
            list1.append(1)
        else:
            list1.append(0)
        if y >= 0:
            eq.append(1)
        else:
            eq.append(0)
        if openprice[i] >= openprice[i + 1]:
            open1.append(0)
        else:
            open1.append(1)

        if closeprice[i]> openprice[i]:
            condition.append(1)
        else:
            condition.append(0)
        if prediction[i] >= prediction[i + 1]:
            predictionprice.append(0)
        else:
            predictionprice.append(1)

    final_list=[]

    for i in range(len(open1)):
        '''
        if list1[i]==predictionprice[i]==condition[i]:
            final_list.append(list1[i])
        elif list1[i] == condition[i]:
            final_list.append(list1[i])
        elif  list1[i]==predictionprice[i]:
            final_list.append(list1[i])

        elif condition[i] == predictionprice[i]:
            final_list.append(condition[i])
        '''
        print( len(eq))
        res = condition[i] + list1[i]+eq[i]+predictionprice[i]
        if (res > 2):
            final_list.append(1)
        else:
            final_list.append(0)


    print(open1)
    print(predictionprice)
    for i in range(len(list1)-1):
        if final_list[i] == open1[i]:
            coun += 1


    print(coun/len(final_list))
    res1=(res/4)*100
    res1= str(float("{0:.2f}".format(res1)))
    return res1,res

def StartARIMAForecasting(Actual, P, D, Q):
    print("train arima")
    model = SARIMAX(Actual, order=(P, D, Q))
    model_fit = model.fit(disp=0)
    prediction = model_fit.forecast()[0]
    return prediction


#creating data


#predict next value





def main_stock (data):
    BASE_DIR =os.path.realpath(__file__) + data;
    reading_data = read_data('/home/reem/stock_site/stock/'+data)
    #Suez-Cement
    #Oriental-Weavers
    #T-M-G-Holding
    #Medinet-Nasr-Housing
    dates = reading_data["Date"].tolist()
    length = floor((len(reading_data) - 120) * (0.8))
    dates = dates[length + 120:]

    close_scaled,open_scaled,high,low,openprice,closeprice,scaler_open,scaler_close= data_cleaning_scaling(reading_data)
    trainingX, testingX, trainingY, testingY = data_splitting(close_scaled,open_scaled)
    t = pd.to_datetime(str(dates[-1]))
    tomorowday = []
    for i in range(3):
        t = t + datetime.timedelta(days=1)
        tomorowday.append(t.strftime('%Y/%m/%d'))
    #GRU_model(trainingX, trainingY)
    pred = GRU_model_load(testingX, testingY,trainingX)
    pred1 = scaler_open.inverse_transform(pred)
    testingy1=scaler_open.inverse_transform(testingY)
    testy=[]
    prediction=[]
    changing=[]
    for i in reversed (testingy1):
        testy.append(i)
    for i in reversed (pred1):
        prediction.append(i)
    for i in range(20):
        changing.append(abs(testy[i]-prediction[i]))
    change=max(changing)
    openprice_split=openprice[length + 120:]
    closeprice_split=closeprice[length + 120:]

    accuracy_up_down,res=classification(high,low,openprice_split,closeprice_split,pred1)
    pred2 ,test=GRU_predict(close_scaled)
    l=[]
    l1=[]
    for i in range(3):
        list_of_pred= scaler_open.inverse_transform(pred2[i])
        if ((res)) > 2:
            l.append(list_of_pred + change)
        else:
            l.append(list_of_pred - change)
        l1.append(l[i][0][0])
        print(list_of_pred)
    keras.backend.clear_session()

    return reading_data ,dates ,prediction,l1,testy,accuracy_up_down