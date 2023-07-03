import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential,load_model
from keras.layers import Dense, LSTM,Dropout
import streamlit as st
import math
import tensorflow as tf

st.title('Stock Trend Predictions')

user_input = st.text_input('Enter Stock Ticker','GBIME')

df = pd.read_csv('Data/'+ user_input+'.csv')

df.drop('Unnamed: 0',axis=1,inplace=True)
df['t'] = pd.to_datetime(df['t']).dt.date
df.set_index(pd.to_datetime(df['t']), inplace=True)
df.drop('t',axis=1,inplace=True)
st.write(df.head())

st.subheader(user_input)

#closing price data
st.subheader('Closing price over time')
fig = plt.figure(figsize=(16,8))
plt.plot(df['c'])
st.pyplot(fig)

st.subheader('Closing Price over time with MA 50')
ma50 = df.c.rolling(50).mean()
fig = plt.figure(figsize=(12,6))
plt.plot(ma50,'r')
plt.plot(df.c)
st.pyplot(fig)

st.subheader('Closing Price over time with MA 100')
ma100 = df.c.rolling(100).mean()
fig = plt.figure(figsize=(12,6))
plt.plot(ma100,'y')
plt.plot(df.c)
st.pyplot(fig)

#dataframe with only close price
data = df.filter(['c'])

#convert the df to a numpy array
dataset = data.values

# get the number of rows to train the model on
training_data_len = math.ceil(len(dataset)*0.8)

#scale the data
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(dataset)

#random seed
tf.random.set_seed(42)

#training the dataset
train_data = scaled_data[0:training_data_len,:]

#----------------------------------------------------------------------------------
#split the data into x_train and y_train 
# x_train = []
# y_train = []

# for i in range(60, len(train_data)):
#     x_train.append(train_data[i-60:i,0])
#     y_train.append(train_data[i,0])

#     # if i<=61:
#     #     print(x_train)
#     #     print(y_train)
#     #     print()

# #convert the x train and y train to numpy arrays
# x_train , y_train = np.array(x_train), np.array(y_train)

# #reshape the data
# x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))

#----------------------------------------------------------------------------------

#load my model
model = load_model('LSTM-model/keras_model.h5')

#create the testing dataset
test_data = scaled_data [training_data_len-60:,:]

#create dataset x_test and y_test
x_test=[]
y_test=dataset[training_data_len:,:]
for i in range(60,len(test_data)):
    x_test.append(test_data[i-60:i,0])

#convert data to a numpy array
x_test = np.array(x_test)

#reshape the data
x_test = np.reshape(x_test,(x_test.shape[0],x_test.shape[1],1))

#get the models predicted price values
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)

#plot the data
train = data[:training_data_len]
valid = data[training_data_len:]
valid['predictions'] = predictions

# final graph visualise
st.subheader('Original vs Prediction')
fig = plt.figure(figsize=(16,8))
plt.title('Model')
plt.xlabel('date',fontsize=18)
plt.ylabel('close price',fontsize=18)
plt.plot(train['c'])
plt.plot(valid[['c','predictions']])
plt.legend(['Train','val','predictions'],loc='lower right')
st.pyplot(fig)