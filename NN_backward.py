'''
Todo:
    1. get weight and bias from training
    2. more comments
    3. more test result
'''

import tensorflow as tf
import numpy as np
import NN_input
import NN_forward
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import seaborn as sns

STEPS = 100000
BATCH_SIZE = 30
LEARNING_RATE_BASE = 0.001
LEANING_RATE_DECAY = 0.999
REGULARIZER = 0.01


def backward():
    '''
    backpropagation function
    learn the weight matrices (w1, w2) 
    '''
    X, Y, sample_num, input_num, output_num, X_test, Y_test =
    NN_input.get_data()
    x = tf.placeholder(tf.float32, shape = (None, input_num))
    y = tf.placeholder(tf.float32, shape = (None, output_num))
    # make FAKE inputs to build the pipeline and add the real input at last

    y_hat, w1, w2, b1, b2 = NN_forward.forward(x, y, input_num, output_num,
    REGULARIZER)
    # estimation of y, and the weight matrix and the using the FAKE variable

    global_step = tf.Variable(0, trainable = False)
    # assign the STEP counter
    learning_rate = tf.train.exponential_decay(
            LEARNING_RATE_BASE,
            global_step,
            sample_num / BATCH_SIZE,
            LEANING_RATE_DECAY,
            staircase = True)
    # set the decay of learning rate

    loss_mse = tf.reduce_mean(tf.square(y_hat - y))
    loss_total = loss_mse + tf.add_n(tf.get_collection('losses'))
    # mean of square error
    # the loss function that will be optimize
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss_total)
    # every train step is to minimize the loss and change parameters by learning rate

    with tf.Session() as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        # initialize all variables
        for i in range(STEPS):
            start = (i * BATCH_SIZE) % sample_num
            end = start + BATCH_SIZE
            # set this step's start and end of training data
            # every time input data with `BATCH_SIZE`
            sess.run(train_step, feed_dict = {x: X[start:end], y: Y[start:end]})
            # run the train_step
            if i % 2000 == 0:
                loss_v = sess.run(loss_total, feed_dict = {x: X, y: Y})
                print('After %d steps, loss is %f' % (i, loss_v))
        y_estimate = sess.run(y_hat, feed_dict = {x: X_test}) 
        W1, W2, B1, B2 = sess.run([w1, w2, b1, b2])
    np.savetxt('./Estimate_y.csv', y_estimate, delimiter = ',')
    np.savetxt('./weights/W1.csv', W1, delimiter = ",")
    np.savetxt('./weights/W2.csv', W2, delimiter = ",")
    np.savetxt('./weights/B1.csv', B1, delimiter = ",")
    np.savetxt('./weights/B2.csv', W2, delimiter = ",")
    error_rate = [abs(Y_test[i] - y_estimate[i]) / Y_test[i] for i in range(Y_test.shape[0])]
    plt.figure()
    sns.boxplot(x = list(range(1, len(error_rate) + 1)), y = error_rate)
    plt.xlabel('Days')
    plt.ylabel('Average Error Rate of Included Companies')
    plt.savefig('./PNGresults/ErrorRate.png')
    for i in range(Y_test.shape[0]):
        plt.figure(figsize = (16, 9))
        plt.scatter(range(output_num), y_estimate[i].tolist(), color = 'red', label = 'Estimate')
        plt.scatter(range(output_num), Y_test[i].tolist(), color = 'blue', label = 'RealData')
        plt.legend()
        plt.xlabel('Companies')
        plt.ylabel('Adj Close')
        plt.title('Test of Estimate Day %d' % i)
        plt.savefig('./PNGresults/TestEstimate_day%d.png' % i)

if __name__ == '__main__':
    backward()
