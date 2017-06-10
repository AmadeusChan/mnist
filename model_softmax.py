import pandas as pd
import numpy as np

# to import data set, in particular trainning set from train.csv and testing set from test.csv
df = pd.read_csv( 'train.csv' )
train = df.values

"""
x_train = train[:,1:]
y_train = train[:,:1]
print x_train.shape
print y_train.shape
print y_train
"""

df = pd.read_csv( 'test.csv' )
x_test = df.values
print x_test.shape
print x_test

# to split trainning set into two sets, one of which is employed to train the model while another is used to validate it for model selection

np.random.shuffle( train )
size_of_validation_set = int( train.shape[0]/3 )

x_valid = train[:size_of_validation_set, 1:]
y_valid = train[:size_of_validation_set, :1]
x_train = train[size_of_validation_set:, 1:]
y_train = train[size_of_validation_set:, :1]

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

print y_valid.shape
print x_train.shape
print y_train.shape
print y_train

"""
z = to_one_hot( y_valid )
print z.shape
"""

y_valid = to_one_hot( y_valid )
y_train = to_one_hot( y_train )

# employing the tensorflow framework to construct the model 

"""
for _ in range(10000):
	batch_xs, batch_ys = get_batch( 100 ) 
	print batch_xs.shape,' ',batch_ys.shape
"""

import tensorflow as tf
x = tf.placeholder( tf.float32, [None, 784] )
W = tf.Variable( tf.zeros( [784, 10] ) )
b = tf.Variable( tf.zeros( [10] ) )
y = tf.nn.softmax( tf.matmul( x, W ) + b )

y_ = tf.placeholder( tf.float32, [None, 10] )
cross_entropy = tf.reduce_mean( - tf.reduce_sum( y_*tf.log(y), reduction_indices=[1] ) )

# to train the model

train_step = tf.train.GradientDescentOptimizer( 0.1 ).minimize( cross_entropy )

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

for _ in range(1000):
	batch_xs, batch_ys = get_batch( 100 ) 
	sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(sess.run(accuracy, feed_dict={x: x_valid, y_: y_valid}))

"""
for _ in range(10000):
	batch_xs, batch_ys = get_batch( 100 ) 
	sess.run( train_step, feed_dict={x: batch_xs, y_:batch_ys} )
	if _%100 ==0:
		correct_prediction = tf.equal( tf.argmax(y,1), tf.argmax(y_,1) )
		accuracy = tf.reduce_mean( tf.cast( correct_prediction, tf.float32 ) )
		training_accuracy = sess.run( accuracy, feed_dict={x: batch_xs, y_: batch_ys} )
		validation_accuracy = sess.run( accuracy, feed_dict={x: x_valid, y_: y_valid} )
		print "step %d, trainning accuracy %g, validation accuracy %g"%(_, training_accuracy, validation_accuracy)
"""


