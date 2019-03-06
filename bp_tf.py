import tensorflow as tf
import numpy as np

Input = tf.placeholder(tf.float32,[None,6])
output = tf.placeholder(tf.float32,[None,4])

def Network(input,input_size,output_size):
	_w1_init = np.array([[.1,.2,.3,.4],
						 [.1, .2, .3,.4],
						 [.1, .2, .3,.4],
						 [.1, .2, .3,.4],
						 [.1, .2, .3,.4],
						 [.1, .2, .3,.4]],dtype=np.float32)
	#_w1_init = tf.random_normal ([input_size, 3])
	_w1 = tf.Variable(name='w1',initial_value=_w1_init)
	_b1 = tf.Variable(name='b1', initial_value=tf.zeros([4]))
	layer1 = tf.add(tf.matmul(input, _w1), _b1)
	layer1o = tf.nn.tanh(layer1)


	_w2_init = np.array([[.2,.2,.3,.1],
						 [.2, .2, .3,.1],
						 [.2, .2, .3,.1],
						 [.2, .2, .3, .1]],dtype=np.float32)
	_w2 = tf.Variable(name='w2', initial_value=_w2_init)
	_b2 = tf.Variable(name='b2', initial_value=tf.zeros([output_size]))

	layer2 = tf.add(tf.matmul(layer1o,_w2), _b2)
	layer2o = tf.nn.tanh(layer2)

	_w3_init = np.array([[.1, .8, .3, .1],
						 [.2, .4, .2, .3],
						 [.3, .6, .5, .1],
						 [.4, .2, .3, .7]], dtype=np.float32)
	_w3 = tf.Variable(name='w3', initial_value=_w3_init)
	_b3 = tf.Variable(name='b3', initial_value=tf.zeros([output_size]))

	layer3 = tf.add(tf.matmul(layer2o, _w3), _b3)
	layer3o = tf.nn.tanh(layer3)

	return layer3o


def train():
	predict = Network(Input,6,4)
	#lb = tf.nn.softmax_cross_entropy_with_logits_v2(labels=output,logits=predict)
	#loss = tf.reduce_mean(lb)#0.5 * (predict - output) ** 2

	lb =  tf.square(predict - output)
	#lb = tf.nn.softmax_cross_entropy_with_logits(labels=output,logits=predict)
	#lb  = tf.reduce_mean(lb)
	#grad = tf.train.GradientDescentOptimizer(learning_rate=0.05).minimize(lb)
	#grad = tf.train.MomentumOptimizer(learning_rate=0.001,momentum=0.9,use_nesterov=True).minimize(lb)
	#grad = tf.train.AdadeltaOptimizer(learning_rate=0.001,rho=0.95).minimize(lb)
	#grad = tf.train.AdagradOptimizer(learning_rate=0.05,initial_accumulator_value=0.1).minimize(lb)
	grad = tf.train.RMSPropOptimizer(learning_rate=0.05,decay=0.99,momentum=0).minimize(lb)
	#grad = tf.train.AdamOptimizer(learning_rate=0.05,beta1=0.9,beta2=0.999).minimize(lb)

	correct_prediction = tf.equal(tf.arg_max(predict, 1), tf.arg_max(output, 1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
	with tf.Session() as session:
		session.run(tf.global_variables_initializer())
		#tx = np.zeros((1,6,1))
		for e in range(100):
			tx = np.array([[.1,.2,.3,.4,.5,.6]])
			ty = np.array([[.4,.5,0,.1]])
			print('WB0->',session.run(tf.trainable_variables()))
			p = session.run([grad,lb,predict],feed_dict={Input:tx,output:ty})
			print(p)
			print('WB1->',session.run(tf.trainable_variables()))
			#variable_name = [v.name for v in tf.trainable_variables()]
			#print(variable_name)

			#print(session.run(predict,feed_dict={Input:np.array([[.1,.2,.3,.4,.5,.6]])}))
			# 取得y得最大概率对应的数组索引来和y_的数组索引对比，如果索引相同，则表示预测正确

			#print(session.run([loss], feed_dict={Input:tx,output:ty}))

if __name__ == '__main__':
	train()



