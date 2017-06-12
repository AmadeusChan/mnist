import pandas as pd
import numpy as np
import os

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

def weight_variable( shape ):
	initial = tf.truncated_normal( shape, stddev=0.1 )
	return tf.Variable( initial )

def bias_variable( shape ):
	initial = tf.constant( 0.1, shape=shape ) 
	return tf.Variable( initial )

def conv2d( x, W, padding ):
	return tf.nn.conv2d( x, W, strides=[1, 1, 1, 1], padding=padding )

def max_pool_2x2( x ):
	return tf.nn.max_pool( x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME' )

def evaluate( accuracy, data_x, data_y ):
	sz = data_x.shape[0]
	size_of_block = 100
	i = 0
	cnt = 0
	while True:
		if i+size_of_block<=sz:
			cnt = cnt + size_of_block * accuracy.eval( feed_dict = { x: data_x[i: i+size_of_block], y_: data_y[ i: i+size_of_block ], keep_prob: 1.0} )
			i = i + size_of_block
		else:
			size_of_block = sz - i
			cnt = cnt + size_of_block * accuracy.eval( feed_dict = { x: data_x[i: i+size_of_block], y_: data_y[ i: i+size_of_block ], keep_prob: 1.0} )
			break
	return cnt*1.0/sz

x = tf.placeholder( tf.float32, [ None, 784 ] )
y_ = tf.placeholder( tf.float32, [ None, 10 ] )
x_image = tf.reshape( x, [ -1, 28, 28, 1 ] )

W_conv1 = weight_variable( [ 5, 5, 1, 32 ] )  
b_conv1 = bias_variable( [32] )

h_conv1 = tf.nn.relu( conv2d( x_image, W_conv1, 'VALID' ) + b_conv1 )
h_pool1 = max_pool_2x2( h_conv1 )

W_conv2 = weight_variable( [ 5, 5, 32, 128 ] ) 
b_conv2 = bias_variable( [128] )

h_conv2 = tf.nn.relu( conv2d( h_pool1, W_conv2, 'SAME' ) + b_conv2 )
h_pool2 = max_pool_2x2( h_conv2 )

W_conv3 = weight_variable( [ 5, 5, 128, 512 ] )
b_conv3 = bias_variable( [512] )

h_conv3 = tf.nn.relu( conv2d( h_pool2, W_conv3, 'SAME' ) + b_conv3 )
h_pool3 = max_pool_2x2( h_conv3 )

keep_prob = tf.placeholder( tf.float32 )

W_fc1 = weight_variable( [ 3*3*512, 4096 ] )
b_fc1 = bias_variable( [4096] )

h_pool3_flat = tf.reshape( h_pool3, [ -1, 3*3*512 ] )
h_fc1 = tf.nn.relu( tf.matmul( h_pool3_flat, W_fc1 ) + b_fc1 )
h_fc1_drop = tf.nn.dropout( h_fc1, keep_prob )

W_fc2 = weight_variable( [ 4096, 1024 ] )
b_fc2 = bias_variable( [1024] )

h_fc2 = tf.nn.relu( tf.matmul( h_fc1_drop, W_fc2 ) + b_fc2 )
h_fc2_drop = tf.nn.dropout( h_fc2, keep_prob )

W_fc3 = weight_variable( [ 1024, 10 ] )
b_fc3 = bias_variable( [10] )

y_conv = tf.matmul( h_fc2_drop, W_fc3 ) + b_fc3

# to employ cross-entropy as the loss function

cross_entropy = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits( labels=y_, logits=y_conv ) )

# to train the model

train_step = tf.train.AdamOptimizer( 1e-4 ).minimize( cross_entropy )

correct_prediction = tf.equal( tf.argmax( y_conv, 1 ), tf.argmax( y_, 1 ) )
accuracy = tf.reduce_mean( tf.cast( correct_prediction, tf.float32 ) )

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.InteractiveSession(config = config)

#sess = tf.InteractiveSession()
sess.run( tf.global_variables_initializer() )

f = file( "result_of_cnn_origin.txt", "a" )
f.write('\n')
f.close()

saver = tf.train.Saver()
model_name = "convolutional_multilayer_network_origin"
highest_valid_accuracy = 0

for _ in range(50000):
	batch_xs, batch_ys = get_batch( 50 ) 
	if _%50==0:
		train_accuracy = evaluate( accuracy, batch_xs, batch_ys )
		valid_accuracy = evaluate( accuracy, x_valid, y_valid )
		print( "step %d, training accuracy %g, validation accuracy %g"%(_, train_accuracy, valid_accuracy) )
		f = file( "result_of_cnn_origin.txt", "a" )
		f.write( "%d %g %g\n"%(_, train_accuracy, valid_accuracy) )
		f.close()
		if valid_accuracy>highest_valid_accuracy:
			print "Highest validation accuracy!"
			saver.save( sess, os.path.join(".", model_name) )
			highest_valid_accuracy=valid_accuracy

	train_step.run( feed_dict={x: batch_xs, y_: batch_ys, keep_prob:0.5} )

saver = tf.train.Saver([W_conv1, b_conv1, W_conv2, b_conv2, W_conv3, b_conv3, W_fc1, b_fc1, W_fc2, b_fc2, W_fc3, b_fc3])
saver.restore(sess, model_name)

y_pred = np.ndarray( shape=(0,10) )

sz = x_test.shape[0]
bl = 300
pt = 0 
while True:
	if pt+bl<=sz:
		y_tmp = sess.run( y_conv, feed_dict = { x: x_test[ pt: pt+bl ], keep_prob: 1.0 } )
		y_pred = np.append( y_pred, y_tmp, 0 )
		pt = pt + bl
	else:
		if pt==sz:
			break
		y_tmp = sess.run( y_conv, feed_dict={ x: x_test[pt: sz], keep_prob:1.0 } )
		y_pred = np.append( y_pred, y_tmp, 0 )
		break

y_test = np.argmax( y_pred, 1 )

f = file( "prediction_by_convolutional_multilayer_network_origin.csv", "w" )
f.write("ImageId,Label\n")
n = y_test.shape[0]
for i in range(n):
	f.write("%d,%d\n"%(i+1, y_test[i]))

f.close()

"""
	sess.run( train_step, feed_dict={x: batch_xs, y_:batch_ys} )
	if _%100 ==0:
		correct_prediction = tf.equal( tf.argmax(y,1), tf.argmax(y_,1) )
		accuracy = tf.reduce_mean( tf.cast( correct_prediction, tf.float32 ) )
		training_accuracy = sess.run( accuracy, feed_dict={x: batch_xs, y_: batch_ys} )
		validation_accuracy = sess.run( accuracy, feed_dict={x: x_valid, y_: y_valid} )
		print "step %d, trainning accuracy %g, validation accuracy %g"%(_, training_accuracy, validation_accuracy)
		f = file( "result_of_cnn_origin.txt", "a" )
		f.write('%d %g %g\n'%(_, training_accuracy, validation_accuracy))
		f.close()
"""


