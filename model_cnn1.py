import pandas as pd
import numpy as np

# to import data set, in particular trainning set from train.csv and testing set from test.csv
df = pd.read_csv( 'train.csv' )
train = df.values

df = pd.read_csv( 'test.csv' )
x_test = df.values

# to split trainning set into two sets, one of which is employed to train the model while another is used to validate it for model selection

np.random.shuffle( train )
size_of_validation_set = int( train.shape[0]/3 )

x_valid = train[:size_of_validation_set, 1:]
y_valid = train[:size_of_validation_set, :1]
x_train = train[size_of_validation_set:, 1:]
y_train = train[size_of_validation_set:, :1]

x_train = x_train / 255.0
x_valid = x_valid / 255.0

def get_batch( sz ):
	batch_x = np.ndarray( shape=(0,784) )
	batch_y = np.ndarray( shape=(0,10) )
	for i in range(sz):
		pt = np.random.randint( 0, x_train.shape[0] )
		batch_x = np.append( batch_x, x_train[pt: pt+1,:], axis=0 )
		batch_y = np.append( batch_y, y_train[pt: pt+1,:], axis=0 )
	return batch_x, batch_y

def to_one_hot( arr ):
	tmp = np.ndarray( shape=( 0, 10 ) )
	cnt = 0 
	sz = arr.shape[0]
	for i in range(sz):
		x = np.ndarray( shape=(1, 10) )
		for j in range(10):
			if j==arr[i]:
				x[0, j]=1
			else:
				x[0, j]=0
		tmp = np.append(tmp, x, axis=0)
	return tmp

y_valid = to_one_hot( y_valid )
y_train = to_one_hot( y_train )

# employing the tensorflow framework to construct the model 

import tensorflow as tf


