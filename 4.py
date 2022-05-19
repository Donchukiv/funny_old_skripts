import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.wrappers.scikit_learn import KerasRegressor
import matplotlib.pyplot as plt

learning_rate = 0.004
decay = 1e-3 / 200
epochs = 300
batch_size = 32
size = 900

dataset = np.linspace(-10,10,1000)
dataset = np.array([dataset, np.sin(dataset)]).T

shuffled_dataset = dataset.copy()
np.random.shuffle(shuffled_dataset)

trainX = shuffled_dataset[:size,:1]
trainY = shuffled_dataset[:size,1]
testX = shuffled_dataset[size:,:1]
testY = shuffled_dataset[size:,1]

def build():
	model = Sequential()
	model.add(Dense(64, input_dim=1, activation='relu'))
	model.add(Dense(64, activation='relu'))
	model.add(Dense(1))
	model.compile(loss='mse', optimizer=Adam(lr=learning_rate, decay=decay))
	return model

model = build()
history = model.fit(trainX, trainY, validation_data=(testX, testY), epochs=epochs, batch_size=batch_size)
plt.subplot(2,1,1)
plt.plot(range(1,epochs+1), history.history['loss'], color='blue', label='train loss')
plt.plot(range(1,epochs+1), history.history['val_loss'], color='orange', label='test loss')
plt.legend()
plt.subplot(2,1,2)
plt.scatter(range(len(dataset)), dataset[:,1], color='green', label='true', alpha=0.3)
plt.scatter(range(len(dataset)), model.predict(dataset[:,:1]), color='red', 
	label='predicted', alpha=0.3)
plt.legend()
plt.show()
