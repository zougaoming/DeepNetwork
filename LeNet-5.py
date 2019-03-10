# coding=utf-8
import tensorflow as tf
from tensorflow.contrib.layers import flatten
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
from sklearn.utils import shuffle

# 定义网络
def LeNet(input_tensor):
    # C1  conv  Input=32*32*1, Output=28*28*6
    conv1_w = tf.Variable(tf.truncated_normal(shape=[5, 5, 1, 6], mean=0, stddev=0.1),name='conv1_w')
    conv1_b = tf.Variable(tf.zeros(6))
    conv1 = tf.add(tf.nn.conv2d(input_tensor, conv1_w, strides=[1, 1, 1, 1], padding='VALID'),conv1_b)
    conv1 = tf.nn.relu(conv1,name='conv1_out')
    #tf.add_to_collection('conv1_out', conv1)

    # S2 Pooling Input=28*28*6 Output=14*14*6
    pool_1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # C3 conv Input=14*14*6 Output=10*10*6
    conv2_w = tf.Variable(tf.truncated_normal(shape=[5, 5, 6, 16], mean=0, stddev=0.1),name='conv2_w')
    conv2_b = tf.Variable(tf.zeros(16))
    conv2 = tf.nn.conv2d(pool_1, conv2_w, strides=[1, 1, 1, 1], padding='VALID')+conv2_b
    conv2 = tf.nn.relu(conv2)

    # S4 Pooling Input=10*10*6 OutPut=5*5*16
    pool_2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # Flatten Input=5*5*16 Output=400
    fc1 = flatten(pool_2)

    # C5 conv Input=5*5*16=400 Output=120
    fc1_w = tf.Variable(tf.truncated_normal(shape=(400, 120), mean=0, stddev=0.1),name='fc1_w')
    fc1_b = tf.Variable(tf.zeros(120))
    fc1 = tf.matmul(fc1, fc1_w) + fc1_b

    # F6 Input=120 OutPut=84
    fc2_w = tf.Variable(tf.truncated_normal(shape=(120, 84), mean=0, stddev=0.1),name='fc2_w')
    fc2_b = tf.Variable(tf.zeros(84))
    fc2 = tf.matmul(fc1, fc2_w)+fc2_b
    fc2 = tf.nn.relu(fc2)

    # F7 Input=84  Output=10
    fc3_w = tf.Variable(tf.truncated_normal(shape=(84, 10), mean=0, stddev=0.1),name='fc3_w')
    fc3_b = tf.Variable(tf.zeros(10))
    logits = tf.matmul(fc2, fc3_w) + fc3_b
    out = {'logits': logits, 'conv1_out': conv1,'test':fc1_w}
    return out

# 参数
BATCH_SIZE = 128
EPOCHS = 100
RATE = 0.1


def printImage(image):
    for i in range(28):
        for j in range(28):
            if image[i][j] == 0 :
                print(' ',end='')
            else:
                print('*',end='')
        print('')

def train(mnist):
    x_train, y_train = mnist.train.images, mnist.train.labels
    x_validation, y_validation = mnist.validation.images, mnist.validation.labels
    printImage(x_train[0])
    # print(x_train[0].shape)
    x_train = np.pad(x_train, ((0, 0), (2, 2), (2, 2), (0, 0)),'constant')
    x_validation = np.pad(x_validation, ((0, 0), (2, 2), (2, 2), (0, 0)),'constant')
    # print(x_train[0].shape)


    x = tf.placeholder(tf.float32, shape=[None, 32, 32, 1])
    y = tf.placeholder(tf.int32, shape=[None, ])
    one_hot_y = tf.one_hot(y, 10)

    Net = LeNet(x)
    y_ = Net['logits']
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=one_hot_y, logits=y_)
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    train_step = tf.train.GradientDescentOptimizer(RATE).minimize(cross_entropy_mean)

    correct_prediction = tf.equal(tf.argmax(y_, 1), tf.argmax(one_hot_y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        for i in range(EPOCHS):
            x_train, y_train = shuffle(x_train, y_train)
            for offset in range(0, 1, BATCH_SIZE):
            #for offset in range(0, len(x_train),BATCH_SIZE):
                end = offset + BATCH_SIZE
                batch_x, batch_y = x_train[offset:end], y_train[offset:end]
                _,p = session.run([train_step,Net], feed_dict={x:batch_x, y:batch_y})
                #print(p['test'])
                #np.set_printoptions(threshold=np.inf)
                #print(batch_x)
                #print(p['conv1_out'])
                #print('WB1->', session.run(tf.trainable_variables('conv1_out')))
                #print(session.run(tf.get_collection('conv1_out')))
                #feature = graph.get_operation_by_name("h_pool_flat").outputs[0]
            print("EPOCHS:", i+1)
            accuracy_score = session.run(accuracy, feed_dict={x:x_validation, y:y_validation})
            print('Validation Accuracy', accuracy_score)
        # test
        x_test, y_test = mnist.test.images, mnist.test.labels
        x_test = np.pad(x_test, ((0, 0), (2, 2), (2, 2), (0, 0)),'constant')
        test_accuracy = session.run(accuracy, feed_dict={x:x_test, y:y_test})
        print('Test Accuracy', test_accuracy) # test_accuracy = 0.9876


def main(argv=None):
    mnist =input_data.read_data_sets("data/mnist/", reshape=False)
    train(mnist)

if __name__ == '__main__':
	tf.app.run()