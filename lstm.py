from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import matplotlib.pyplot as plt
import numpy
import math
import os
script_dir = os.path.dirname(__file__)
os.chdir(script_dir)

dataframe = pd.read_csv('airline-passengers.csv', usecols = [1], engine = 'python')
dataframe.plot(y = 'Passengers')
print(dataframe)
plt.show()

dataset = dataframe.values
dataset = dataset.astype('float32')
scaler = MinMaxScaler(feature_range = (0, 1))
dataset = scaler.fit_transform(dataset)

batch_size = 1
look_back = 5
skip = 1
hidden_size = 128
num_epochs = 10
TRAIN_PERCENT = 0.67

numpy.random.seed(7)

train_size = int(len(dataset) * TRAIN_PERCENT)
test_size = len(dataset) - train_size
train, test  = dataset[:train_size], dataset[train_size:]

def create_dataset(dataset, look_back=look_back):
    dataX, dataY=[], []

    for i in range(len(dataset) - look_back - 1):
        a=dataset[i:(i+look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return numpy.array(dataX), numpy.array(dataY)

trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)

trainX = numpy.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

model = Sequential()
model.add(LSTM(hidden_size, input_shape=(1, look_back)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.summary()

model.fit(trainX, trainY, epochs=num_epochs, batch_size=batch_size, verbose=2)

trainPredict=model.predict(trainX)
testPredict = model.predict(testX)

print("\nTraining prediction:\n", trainPredict[:10])
print("\nTesting prediction:\n", testPredict[:10])

trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])

print("Training Actual:\n", trainY[0,:10])
print("Training Predictions:\n", trainPredict[:10])
print("Testing Actual:\n", testY[0,:10])
print("Testing Predictions:\n", testPredict[:10])

# Actual: A single row of many labels. Prediction: Many rows with a single value

trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
print('Train Score: %.2f RMSE' % (trainScore))

testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:, 0]))
print('Test Score: %.2f RMSE' % (testScore))