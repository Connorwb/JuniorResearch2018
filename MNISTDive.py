# Plenty of help from sentdex on YouTube.
import tensorflow as tf
import time
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
n_nodes_hl1 = 20
n_nodes_hl2 = 20
classes = 10
batch_size = 100
x = tf.placeholder('float', [None, 784])  # input
y = tf.placeholder('float')
corruption_level = 0.00
def neural_network_model(data):
    tf.set_random_seed(24)
    hidden_1_layer = {'weights': tf.Variable(tf.random_normal([784, n_nodes_hl1])),
                      'biases': tf.Variable(tf.random_normal([n_nodes_hl1]))}
    hidden_2_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
                      'biases': tf.Variable(tf.random_normal([n_nodes_hl2]))}
    output_l = {'weights': tf.Variable(tf.random_normal([n_nodes_hl2, classes])),
                'biases': tf.Variable(tf.random_normal([classes]))}
    l1 = tf.add(tf.matmul(data, hidden_1_layer['weights']), hidden_1_layer['biases'])
    l1 = tf.nn.sigmoid(l1)
    l2 = tf.add(tf.matmul(l1, hidden_2_layer['weights']), hidden_2_layer['biases'])
    l2 = tf.nn.sigmoid(l2)
    output = (tf.matmul(l2, output_l['weights']) + output_l['biases'])
    return output
def training(x):
    prediction = neural_network_model(x)
    cost = tf.losses.mean_squared_error(predictions=prediction, labels=y )
    optimizer = tf.train.MomentumOptimizer(learning_rate=.01, momentum=0.5).minimize(cost)
    print("beginning training!")
    starttime = time.time()
    d_epochs = 100
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        #print(mnist.train.labels[43][8])
        to_corrupt = corruption_level * len(mnist.train.labels)
        if to_corrupt != 0:
            every_x = int(len(mnist.train.labels) / to_corrupt)
            ii = every_x
            current_dis = 0
            while ii < len(mnist.train.labels):
                to_null = 10
                iii = 0
                while iii < 10:
                    if mnist.train.labels[ii][iii] == 1.0:
                        to_null = iii
                    iii += 1
                if to_null == current_dis :
                    current_dis += 1
                if current_dis == 10 :
                    current_dis=0
                mnist.train.labels[ii][to_null] = 0.0
                mnist.train.labels[ii][current_dis] = 1.0
                ii += every_x
        '''correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        acc = tf.reduce_mean(tf.cast(correct, 'float'))
        print('start accuracy:', acc.eval({x: mnist.test.images, y: mnist.test.labels}))'''
        for epoch in range(d_epochs):
            epoch_l = 0
            '''for _ in range(int(mnist.train.num_examples / batch_size)):
                epoch_x, epoch_y = mnist.train.next_batch(batch_size)
                _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})  # I don't get this line
                epoch_l += c'''
            i = 0
            while i < len(mnist.train.images):
                start = i
                end = i+batch_size
                if end >= len(mnist.train.images):
                    end = len(mnist.train.images)
                epoch_x = mnist.train.images[start:end]
                epoch_y = mnist.train.labels[start:end]
                _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})  # I don't get this line
                epoch_l += c
                i += batch_size
            print_thing = 'Epoch ' + repr(epoch) + '/' + repr(d_epochs) + ' completed. loss:' + repr(epoch_l) + ' '
            print(print_thing)
        endtime = time.time()
        time_elapsed= endtime - starttime
        print(time_elapsed)
        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        acc = tf.reduce_mean(tf.cast(correct, 'float'))
        print('accuracy:', acc.eval({x: mnist.test.images, y: mnist.test.labels}))
training(x)
