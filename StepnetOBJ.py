import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
class Stepnet:
    def __init__(self, n_nodes_hl1, n_nodes_hl2, batch_size, corruption_level):
        mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
        classes = 10
        x = tf.placeholder('float', [None, 784])  # input
        y = tf.placeholder('float')
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
            cost = tf.losses.mean_squared_error(predictions=prediction, labels=y)
            optimizer = tf.train.MomentumOptimizer(learning_rate=.01, momentum=0.5).minimize(cost)
            print("beginning training!")
            d_epochs = 55
            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
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
                        if to_null == current_dis:
                            current_dis += 1
                        if current_dis == 10:
                            current_dis = 0
                        mnist.train.labels[ii][to_null] = 0.0
                        mnist.train.labels[ii][current_dis] = 1.0
                        ii += every_x
                s = 0
                while s < d_epochs:
                    epoch_l = 0
                    i = 0
                    while i < len(mnist.train.images):
                        start = i
                        end = i + batch_size
                        if end >= len(mnist.train.images):
                            end = len(mnist.train.images)
                        epoch_x = mnist.train.images[start:end]
                        epoch_y = mnist.train.labels[start:end]
                        _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})  # I don't get this line
                        epoch_l += c
                        i += batch_size
                        s += 1
                        print_thing = 'Step ' + repr(s) + '/' + repr(d_epochs) + ' completed. loss:' + repr(
                            epoch_l) + ' '
                        print(print_thing)
                correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
                acc = tf.reduce_mean(tf.cast(correct, 'float'))
                print('accuracy:', acc.eval({x: mnist.test.images, y: mnist.test.labels}))
        training(x)
Stepnet(20, 20, 1000, 0.1)
Stepnet(20, 20, 1000, 0.1)
