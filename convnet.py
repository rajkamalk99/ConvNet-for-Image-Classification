import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
import os 


data = input_data.read_data_sets("data/fashion", one_hot=True)
# shapes of training data 
print("Shape of the training data is {}".format(data.train.images.shape))
print("Shape of the training data(labels) is {}".format(data.train.labels.shape))
# shapes of testing data
print("Shape of the testing data is {}".format(data.test.images.shape))
print("Shape of the testing data(labels) is {}".format(data.test.labels.shape))

# reshapeing the data 

train_x = data.train.images.reshape(-1,28,28,1)
test_x = data.test.images.reshape(-1,28,28,1)

train_y = data.train.labels
test_y = data.test.labels

# printing shapes after reshaping the data 

# shapes of training data 
print("Shape of the training data after reshaping is {}".format(train_x.shape))
print("Shape of the training data(labels) after reshaping is {}".format(train_y.shape))
# shapes of testing data
print("Shape of the testing data after reshaping is {}".format(test_x.shape))
print("Shape of the testing data(labels) after reshaping is {}".format(test_y.shape))


# deep neural network

# input data(28*28*1) --> convolution(32-3*3 filters) --> max_pooling(2*2) --> convolution(64-3*3 filters) --> max_pooling(2*2) -->

# convolution(128-3*3 filters) --> max_pooling(2*2) --> Flatten --> Deanse_layer(128 units) --> output_layer(10 units)


training_iters = 200
learning_rate = 0.001
batch_size = 128

n_input = 28
n_classes = 10

x = tf.placeholder("float", [None, 28, 28, 1])
y = tf.placeholder("float", [None, n_classes])


def conv2d(x, w, b, strides=1):
    # Conv2D wrapper, with bias and relu activation

    x = tf.nn.conv2d(x, w, strides=[1 ,strides, strides, 1], padding="SAME")
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)


def maxPool(x, k=2):
    return tf.nn.max_pool(x, ksize=[1,k,k,1], strides=[1,k,k,1], padding="SAME")
     
weights = {

    # the shape is of the form filter_size x no_channels x no_filters 
    "wc1": tf.get_variable("w0", shape=(3,3,1,32), initializer=tf.contrib.layers.xavier_initializer()),
    # in the second layer the no_channels is same as the no_filters of the previous layer
    "wc2": tf.get_variable("w1", shape=(3,3,32,64), initializer=tf.contrib.layers.xavier_initializer()),
    "wc3": tf.get_variable("w2", shape=(3,3,64,128), initializer=tf.contrib.layers.xavier_initializer()),
    # after applying three convolution layers we are transforming a 28*28*1 image into 4*4*1 image so multiply it by the no_filters in the previous layer
    # and the second argument is the no_neurons we want in that layer
    "wd1": tf.get_variable("w3", shape=(4*4*128,128), initializer=tf.contrib.layers.xavier_initializer()),
    "out": tf.get_variable("w4", shape=(128, n_classes), initializer=tf.contrib.layers.xavier_initializer())
}


     
biases = {

    "bc1": tf.get_variable("b0", shape=(32), initializer=tf.contrib.layers.xavier_initializer()),
    "bc2": tf.get_variable("b1", shape=(64), initializer=tf.contrib.layers.xavier_initializer()),
    "bc3": tf.get_variable("b2", shape=(128), initializer=tf.contrib.layers.xavier_initializer()),
    "bd1": tf.get_variable("b3", shape=(128), initializer=tf.contrib.layers.xavier_initializer()),
    "out": tf.get_variable("b4", shape=(10), initializer=tf.contrib.layers.xavier_initializer())
}

def convnet(x, weights, biases):
    # layer one of convolution and max pooling
    conv1 = conv2d(x, weights["wc1"], biases["bc1"])
    conv1 = maxPool(conv1)

    # layer two of convolution and max pooling
    conv2 = conv2d(conv1, weights["wc2"], biases["bc2"])
    conv2 = maxPool(conv2)

    # layer three of convolution and max pooling
    conv3 = conv2d(conv2, weights["wc3"], biases["bc3"])
    conv3 = maxPool(conv3)

    # fully connected layer 
    # reshape conv3 to fit directly into the dense layer(flattening)   

    full = tf.reshape(conv3, [-1, weights["wd1"].get_shape().as_list()[0]])
    full = tf.add(tf.matmul(full, weights["wd1"]), biases["bd1"])
    full = tf.nn.relu(full)

    # now passing this output from fully connected layer to the output layer

    output = tf.add(tf.matmul(full, weights["out"]), biases["out"])

    return output


pred = convnet(x, weights, biases)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))

accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

init = tf.global_variables_initializer()


with tf.Session() as sess:
    sess.run(init)
    train_loss = []
    test_loss = []
    train_accuracy = []
    test_accuracy = []
    summary_writer = tf.summary.FileWriter('./output', sess.graph)

    for iteration in range(training_iters):
        for batch in range(len(train_x)//batch_size):
            batch_x = train_x[batch*batch_size:min(batch_size*(batch+1), len(train_x))]
            batch_y = train_y[batch*batch_size:min(batch_size*(batch+1), len(train_y))]

            opt = sess.run(optimizer, feed_dict={x:batch_x, y:batch_y})

            loss, acc = sess.run([cost, accuracy], feed_dict={x:batch_x, y:batch_y})

        print("Iteration" + str(iteration) + ", Loss = " + 
                            "{:.6f}".format(loss) + ", Training accuracy= " + 
                                "{:.5f}".format(acc))
        print("Optimization Finished")

        test_acc, valid_loss = sess.run([accuracy, cost], feed_dict={x: test_x, y:test_y})
        train_accuracy.append(acc)
        train_loss.append(loss)
        test_accuracy.append(test_acc)
        test_loss.append(valid_loss)
        print("Testing Accuracy :","{:.5f}".format(test_acc))
    summary_writer.close()

