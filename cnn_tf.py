import tensorflow as tf
import numpy as np
from tensorflow.contrib.layers import flatten

Input = tf.placeholder(tf.float64,[None,8,8,1])
output = tf.placeholder(tf.float64,[None,1,1,1])

def Network(input,input_size,output_size):
	_w1_init = np.array([[[[.1]],[[.2]],[[.4]]],
						 [[[.3]], [[.6]],[[.7]]],
						 [[[.5]], [[.2]], [[.8]]]],dtype=np.float64)

	_w1 = tf.Variable(name='w1',initial_value=_w1_init)
	_b1 = tf.Variable(name='b1', initial_value=.0,dtype=np.float64)
	layer1 = tf.add(tf.nn.conv2d(input,_w1,strides=[1, 1, 1, 1],padding="VALID"),_b1)
	layer1o = tf.nn.tanh(layer1)

	_output = tf.nn.pool(layer1o, window_shape=[2, 2], strides=[2, 2], pooling_type='MAX', padding='VALID')  # AVG MAX

	_w1_init2 = np.array([[[[.1]], [[.2]]],
						 [[[.3]], [[.6]]]], dtype=np.float64)

	test  = flatten(_output)
	_w2 = tf.Variable(name='w2', initial_value=_w1_init2)
	_b2 = tf.Variable(name='b2', initial_value=.0,dtype=np.float64)
	layer2 = tf.add(tf.nn.conv2d(_output, _w2, strides=[1, 1, 1, 1], padding="VALID"), _b2)
	layer2o = tf.nn.tanh(layer2)

	output = tf.nn.pool(layer2o,    window_shape=[2,2],strides=[2,2],pooling_type='MAX',padding='VALID')#AVG MAX

	out = {'test':test,'output':output}
	return out


def train():
	out = Network(Input,8,1)
	predict = out['output']
	#lb = tf.nn.softmax_cross_entropy_with_logits_v2(labels=output,logits=predict)
	#loss = tf.reduce_mean(lb)#0.5 * (predict - output) ** 2

	lb = tf.square(predict - output)
	loss = tf.reduce_mean(lb)
	optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.05)
	grad = optimizer.minimize(lb)
	#correct_prediction = tf.equal(tf.arg_max(predict, 1), tf.arg_max(output, 1))
	#accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
	with tf.Session() as session:
		session.run(tf.global_variables_initializer())
		#tx = np.zeros((1,6,1))
		for e in range(100):
			print('-----------',e)
			tx = np.array([[[[.1],[.1],[.9],[.4],[.5],[.6],[.1],[.5]],
						   [[.1], [.2], [.8], [.7], [.0], [.6], [.7], [.1]],
						   [[.4], [.7], [.6], [.4], [.5], [.6], [.3], [.5]],
						   [[.1], [.2], [.3], [.8], [.7], [.8], [.1], [.7]],
						   [[.1], [.9], [.8], [.4], [.5], [.6], [.6], [.5]],
						   [[.5], [.2], [.3], [.9], [.5], [.0], [.1], [.0]],
						   [[.1], [.1], [.9], [.4], [.5], [.3], [.8], [.5]],
						   [[.1], [.2], [.2], [.4], [.8], [.6], [.1], [.5]]]])
			ty = np.array([[[[.4],[.5],[0],[.1],[.6],[.0]],
						   [[.3], [.5], [0], [.1], [.2],[.1]],
						   [[.4], [.4], [.7], [.1], [.2],[.3]],
						   [[.4], [.5], [0], [.2], [.2],[.9]],
						   [[.4], [.5], [0], [.1], [.2],[.5]],
							[[.4], [.5], [0], [.1], [.2],[.6]]]])
			ty2 = np.array([[[[.4], [.5],[.2]],
							 [[0], [.1],[.2]],
							[[.1],[.3],[.2]]
							 ]])
			ty2 = [[[[0.6]]]]
			#print('WB0->',session.run(tf.trainable_variables()))
			_,p = session.run([grad,out],feed_dict={Input:tx,output:ty2})
			print(p['test'])
			#print('WB1->',session.run(tf.trainable_variables()))
			#variable_name = [v.name for v in tf.trainable_variables()]
			#print(variable_name)

			#print(session.run(predict,feed_dict={Input:np.array([[.1,.2,.3,.4,.5,.6]])}))
			# 取得y得最大概率对应的数组索引来和y_的数组索引对比，如果索引相同，则表示预测正确

			#print(session.run([loss], feed_dict={Input:tx,output:ty}))

if __name__ == '__main__':
	train()



