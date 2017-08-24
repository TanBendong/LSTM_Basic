import pandas as pd
import numpy as np

#Read CSV with , as separator and first row as header
df = pd.read_csv('./sunspot.csv',sep = ',',header=1,index_col =0, names =["date","sunspots"] )
df.fillna(0)

spots = df['sunspots'].values

print (len(spots))

#print (spots[0])
#print (spots[len(spots)-2])

def count_nans(x):
   a=0
   for i in range(len(x)):
      if(np.isnan(x[i])):
         a=a+1
   return a 

#print (count_nans(spots))
spots=np.nan_to_num(spots)
#print (count_nans(spots))


#Dataset is ready cleaned -- Visualize
import matplotlib.pyplot as plt
plt.plot(spots)
plt.show()

np.random.seed(7)
spots = spots.astype('float32')

train_len = int(len(spots)*0.8)
test_len  = int(len(spots)*0.2)

training_set = spots[0:train_len]
test_set     = spots[train_len:test_len]
print (len(training_set), len(test_set))

train_X= training_set[0:train_len-1]
train_Y= training_set[1:train_len]
test_X=test_set[0:len(test_set)-1]
test_Y=test_set[1:len(test_set)]

# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back-1):
		a = dataset[i:(i+look_back)]
		dataX.append(a)
		dataY.append(dataset[i + look_back])
	return np.array(dataX), np.array(dataY)
3
4
	
# reshape into X=t and Y=t+1
look_back = 1
#train_X, train_Y = create_dataset(training_set, look_back)
#test_X, test_Y = create_dataset(test_set, look_back)

# reshape into X=t and Y=t+1
#look_back = 1
#trainX, trainY = create_dataset(train, look_back)
#testX, testY = create_dataset(test, look_back)
# reshape input to be [samples, time steps, features]
train_X = np.reshape(train_X, (train_X.shape[0], 1))
test_X = np.reshape(test_X, (test_X.shape[0], 1))

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM


model = Sequential()
model.add(LSTM(3, input_shape=(1,1)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(train_X,train_Y , nb_epoch=100, batch_size=1, verbose=2)

# make predictions
trainPredict = model.predict(train_X)
testPredict = model.predict(test_X)

# calculate root mean squared error
trainScore = math.sqrt(mean_squared_error(train_Y[0], trainPredict[:,0]))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(test_Y[0], testPredict[:,0]))
print('Test Score: %.2f RMSE' % (testScore))

