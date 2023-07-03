from flask import Flask, request, jsonify, render_template
from keras.models import load_model
import json as json
from flask_cors import CORS, cross_origin

import math
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'




@app.route("/", methods=['GET'])
@cross_origin()
def hello_world():
    #loads model
    model = load_model(r'C:\Users\97798\Downloads\Stock Vision\Stock prediction\LSTM-model\adbl_model.h5')

    df = pd.read_csv('./Data/ADBL.csv')

    #dataframe with only close price
    data = df.filter(['c'])

    df.drop('Unnamed: 0',axis=1,inplace=True)
    df['t'] = pd.to_datetime(df['t']).dt.date
    df = df.set_index('t')
    df = df[['c']].copy()

    #convert the df to a numpy array
    dataset = data.values

    # get the number of rows to train the model on

    training_data_len = math.ceil(len(dataset)*0.8)


    #scale the data
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(dataset)


    #training the dataset
    train_data = scaled_data[0:training_data_len,:]

    #split the data into x_train and y_train 
    x_train = []
    y_train = []

    for i in range(60, len(train_data)):
        x_train.append(train_data[i-60:i,0])
        y_train.append(train_data[i,0])

    x_train , y_train = np.array(x_train), np.array(y_train)

    #create the testing dataset
    test_data = scaled_data [training_data_len-60:,:]

    #create dataset x_test and y_test
    x_test=[]
    y_test =dataset[training_data_len:,:]
    for i in range(60,len(test_data)):
        x_test.append(test_data[i-60:i,0])

    #convert data to a numpy array
    x_test = np.array(x_test)

    x_test = np.reshape(x_test,(x_test.shape[0],x_test.shape[1],1))

    #get the models predicted price values
    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions)

    rmse = np.sqrt(np.mean(predictions - y_test)**2)
    rmse

    #plot the data
    train = data[:training_data_len]
    valid = data[training_data_len:]
    valid['predictions'] = predictions

    train = train.to_json()
    valid = valid.to_json()
    predictions = predictions.flatten().tolist()

    date =df.index #.to_json()

    data = [
        {
        "predictions": predictions,
        "train": train,
        "valid": valid 
        }
    ]
    
    return  data

if __name__ == '__main__':
    app.debug = True
    app.run (host = "localhost", port = 5000)