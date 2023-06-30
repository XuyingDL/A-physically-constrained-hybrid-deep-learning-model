import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import h5py
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
import tensorflow as tf
if type(tf.contrib) != type(tf): tf.contrib._warning = None
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import tensorflow as tf
import scipy.io as scio
import scipy.io as sio
from tensorflow.python.framework import ops
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def convert_to_one_hot(Y, C):
    Y = np.eye(C)[Y.reshape(-1)].T
    return Y

def random_mini_batches_GCN_W(X, X1, Y,  W, L, mini_batch_size, seed):
    m = X.shape[0]
    mini_batches = []
    np.random.seed(seed)
    # W = W.T

    permutation = list(np.random.permutation(m))
    shuffled_X = X[permutation, :]
    shuffled_X1 = X1[permutation, :]
    shuffled_Y = Y[permutation, :]
    shuffled_Y = shuffled_Y.reshape((m, Y.shape[1]))
    shuffled_W = W[permutation, :]
    shuffled_L1 = L[permutation, :].reshape((L.shape[0], L.shape[1]), order="F")
    shuffled_L = shuffled_L1[:, permutation].reshape((L.shape[0], L.shape[1]), order="F")

    num_complete_minibatches = math.floor(m / mini_batch_size)

    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[k * mini_batch_size: k * mini_batch_size + mini_batch_size, :]
        mini_batch_X1 = shuffled_X1[k * mini_batch_size: k * mini_batch_size + mini_batch_size, :]
        mini_batch_Y = shuffled_Y[k * mini_batch_size: k * mini_batch_size + mini_batch_size, :]
        mini_batch_W = shuffled_W[k * mini_batch_size: k * mini_batch_size + mini_batch_size, :]
        mini_batch_L = shuffled_L[k * mini_batch_size: k * mini_batch_size + mini_batch_size,
                       k * mini_batch_size: k * mini_batch_size + mini_batch_size]
        mini_batch = (mini_batch_X,mini_batch_X1, mini_batch_Y, mini_batch_W, mini_batch_L)
        mini_batches.append(mini_batch)
    mini_batch = (X, X1, Y, W, L)
    mini_batches.append(mini_batch)

    return mini_batches

def create_placeholders(n_x, n_x1, n_y, n_w):
    isTraining = tf.placeholder_with_default(True, shape=())
    x_in = tf.placeholder(tf.float32, [None, n_x], name="x_in")
    x_in1 = tf.placeholder(tf.float32, [None, n_x1], name="x_in1")
    y_in = tf.placeholder(tf.float32, [None, n_y], name="y_in")
    w_in = tf.placeholder(tf.float32, [None, n_w], name="w_in")
    lap_train = tf.placeholder(tf.float32, [None, None], name="lap_train")

    return x_in, x_in1, y_in, w_in, lap_train, isTraining


def initialize_parameters():
    tf.set_random_seed(1)

    x_w1 = tf.get_variable("x_w1", [39, 64], initializer=tf.contrib.layers.xavier_initializer(seed=1))
    x_b1 = tf.get_variable("x_b1", [64], initializer=tf.zeros_initializer())

    x_jw1 = tf.get_variable("x_jw1", [64 + 64, 64], initializer=tf.contrib.layers.xavier_initializer(seed=1))
    x_jb1 = tf.get_variable("x_jb1", [64], initializer=tf.zeros_initializer())

    x_jw2 = tf.get_variable("x_jw2", [64, 2], initializer=tf.contrib.layers.xavier_initializer(seed=1))
    x_jb2 = tf.get_variable("x_jb2", [2], initializer=tf.zeros_initializer())

    x_conv1_w1 = tf.get_variable("x_conv_w1", [3, 39, 128], initializer=tf.contrib.layers.xavier_initializer(seed=1))
    x_conv1_b1 = tf.get_variable("x_conv_b1", [128], initializer=tf.zeros_initializer())

    x_conv1_w2 = tf.get_variable("x_conv_w2", [3, 128, 256], initializer=tf.contrib.layers.xavier_initializer(seed=1))
    x_conv1_b2 = tf.get_variable("x_conv_b2", [256], initializer=tf.zeros_initializer())

    x_conv1_w3 = tf.get_variable("x_conv_w3", [3,256,64], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
    x_conv1_b3 = tf.get_variable("x_conv_b3", [64], initializer = tf.zeros_initializer())

    parameters = {"x_w1": x_w1,
                  "x_b1": x_b1,
                  "x_jw1": x_jw1,
                  "x_jb1": x_jb1,
                  "x_jw2": x_jw2,
                  "x_jb2": x_jb2,
                  "x_conv1_w1": x_conv1_w1,
                  "x_conv1_b1": x_conv1_b1,
                  "x_conv1_w2": x_conv1_w2,
                  "x_conv1_b2": x_conv1_b2,
                  "x_conv1_w3": x_conv1_w3,
                  "x_conv1_b3": x_conv1_b3
                  }

    return parameters


def GCN_layer(x_in, L_, weights):
    x_mid = tf.matmul(x_in, weights)
    x_out = tf.matmul(L_, x_mid)

    return x_out


def mynetwork(x, x1, parameters, Lap, isTraining, momentums=0.9):
    x1 = tf.reshape(x1, [-1, 1, 39], name="x1")

    with tf.name_scope("x_layer_1"):
        x_z1_bn = tf.layers.batch_normalization(x, momentum=momentums, training=isTraining)
        x_z1 = GCN_layer(x_z1_bn, Lap, parameters['x_w1']) + parameters['x_b1']
        x_z1_bn = tf.layers.batch_normalization(x_z1, momentum=momentums, training=isTraining)
        x_a1 = tf.nn.relu(x_z1_bn)

        x_conv1_z1 = tf.nn.conv1d(x1, parameters['x_conv1_w1'], stride=[1, 1, 1], padding='SAME') + parameters[
            'x_conv1_b1']
        x_conv1_z1_bn = tf.layers.batch_normalization(x_conv1_z1, momentum=momentums, training=isTraining)
        x_conv1_z1_po = tf.layers.max_pooling1d(x_conv1_z1_bn, 2, 2, padding='SAME')
        x_conv1_a1 = tf.nn.relu(x_conv1_z1_po)

    with tf.name_scope("x_layer_2"):

        x_conv1_z2 = tf.nn.conv1d(x_conv1_a1, parameters['x_conv1_w2'], stride=[1, 1, 1], padding='SAME') + parameters['x_conv1_b2']
        x_conv1_z2_po = tf.layers.max_pooling1d(x_conv1_z2, 2, 2, padding='SAME')
        x_conv1_a2 = tf.nn.relu(x_conv1_z2_po)

    with tf.name_scope("x_layer_3"):

        x_conv1_z3 = tf.nn.conv1d(x_conv1_a2, parameters['x_conv1_w3'], stride=[1, 1, 1], padding='SAME') + parameters['x_conv1_b3']
        x_conv1_z3_po = tf.layers.max_pooling1d(x_conv1_z3, 2, 2, padding='SAME')
        x_conv1_a3 = tf.nn.relu(x_conv1_z3_po)

        x_conv1_a3_shape = x_conv1_a3.get_shape().as_list()
        x_conv1_z3_2d = tf.reshape(x_conv1_a3, [-1, x_conv1_a3_shape[1] * x_conv1_a3_shape[2]])

        joint_encoder_layer = tf.concat([x_a1, x_conv1_z3_2d], 1)

    with tf.name_scope("x_joint_layer_1"):
        x_zj1 = tf.matmul(joint_encoder_layer, parameters['x_jw1']) + parameters['x_jb1']
        x_aj1 = tf.nn.relu(x_zj1)

    with tf.name_scope("x_layer_4"):
        x_zj2 = tf.matmul(x_aj1, parameters['x_jw2']) + parameters['x_jb2']
        x_aj2 = tf.nn.softmax(x_zj2)

    l2_loss = tf.nn.l2_loss(parameters['x_w1']) + tf.nn.l2_loss(parameters['x_jw1']) + tf.nn.l2_loss(parameters['x_jw2'])\
              + tf.nn.l2_loss(parameters['x_conv1_w1']) + tf.nn.l2_loss(parameters['x_conv1_w2'])+ tf.nn.l2_loss(parameters['x_conv1_w3'])

    return x_aj2, l2_loss


def mynetwork_optimaization(y_est, y_re, w, l2_loss, reg, learning_rate, global_step):
    y_re = tf.squeeze(y_re, name='y_re')

    with tf.name_scope("cost"):
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_est, labels=y_re)) + reg * l2_loss + tf.reduce_mean(tf.square(y_est[:,1] - w))

    with tf.name_scope("optimization"):
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost, global_step=global_step)
        optimizer = tf.group([optimizer, update_ops])

    return cost, optimizer


def train_mynetwork(x_train, x_val, x_test, y_train, y_val, L_train, L_val, L_test, train1D_x, val1D_x, test1D_x, train_w, val_w,learning_rate_base=0.0001, beta_reg=0.001, num_epochs=300, minibatch_size=16, print_cost=True):
    ops.reset_default_graph()
    tf.set_random_seed(1)
    seed = 1
    (m, n_x) = x_train.shape
    (m, n_y) = y_train.shape
    (m, n_w) = train_w.shape
    (m, n_x1) = train1D_x.shape

    costs = []
    costs_dev = []
    train_acc = []
    val_acc = []
    # x_in:GCN, x_in1:1DCNN
    x_in, x_in1, y_in, w_in, lap_train, isTraining = create_placeholders(n_x, n_x1, n_y, n_w)

    parameters = initialize_parameters()

    with tf.name_scope("network"):
        x_out, l2_loss = mynetwork(x_in, x_in1, parameters, lap_train, isTraining)

    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(learning_rate_base, global_step, 50 * m / minibatch_size, 0.5,staircase=True)

    with tf.name_scope("optimization"):
        cost, optimizer = mynetwork_optimaization(x_out, y_in, w_in, l2_loss, beta_reg, learning_rate, global_step)

    with tf.name_scope("metrics"):
        joint_layerT = tf.transpose(x_out)
        yT = tf.transpose(y_in)
        correct_prediction = tf.equal(tf.argmax(joint_layerT), tf.argmax(yT))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    init = tf.global_variables_initializer()

    with tf.Session() as sess:

        sess.run(init)
        max_acc = 0.985

        # Do the training loop
        for epoch in range(num_epochs + 1):
            epoch_cost = 0.  # Defines a cost related to an epoch
            epoch_acc = 0.
            num_minibatches = int(m / minibatch_size)  # number of minibatches of size minibatch_size in the train set
            seed = seed + 1
            minibatches = random_mini_batches_GCN_W(x_train, train1D_x, y_train, train_w, L_train, minibatch_size, seed)
            for minibatch in minibatches:
                # Select a minibatch
                (batch_x, batch_x1, batch_y, batch_w, batch_l) = minibatch
                # IMPORTANT: The line that runs the graph on a minibatch.
                # Run the session to execute the "optimizer" and the "cost", the feedict should contain a minibatch for (X,Y).
                _, minibatch_cost, minibatch_acc = sess.run([optimizer, cost, accuracy],feed_dict={x_in: batch_x, x_in1: batch_x1, y_in: batch_y,w_in: batch_w, lap_train: batch_l,isTraining: True})

                epoch_cost += minibatch_cost / (num_minibatches + 1)
                epoch_acc += minibatch_acc / (num_minibatches + 1)

            if print_cost == True and (epoch) % 1 == 0:
                features, epoch_cost_dev, epoch_acc_dev = sess.run([x_out, cost, accuracy],feed_dict={x_in: x_val, x_in1: val1D_x, y_in: y_val,w_in: val_w,lap_train: L_val, isTraining: False})
                print("epoch %i: Train_loss: %f, Val_loss: %f, Train_acc: %f, Val_acc: %f" % (
                epoch, epoch_cost, epoch_cost_dev, epoch_acc, epoch_acc_dev))
                if epoch_acc_dev >= max_acc:
                    max_acc = epoch_acc_dev
                    features_out = sess.run([x_out], feed_dict={x_in: x_test, x_in1: test1D_x, lap_train: L_test,isTraining: False})
                    print("test!!! max_acc %f" % (max_acc))

            if print_cost == True and epoch % 1 == 0:
                costs.append(epoch_cost)
                train_acc.append(epoch_acc)
                costs_dev.append(epoch_cost_dev)
                val_acc.append(epoch_acc_dev)

        # lets save the parameters in a variable
        parameters = sess.run(parameters)
        print("Parameters have been trained!")

        return parameters, costs, costs_dev, train_acc, val_acc, features_out


Train_X = scio.loadmat('*/miniGCN/Train_X.mat')
TrLabel = scio.loadmat('*/miniGCN/TrLabel.mat')
Val_X = scio.loadmat('*/miniGCN/Val_X.mat')
ValLabel = scio.loadmat('*/miniGCN/ValLabel.mat')
Test_X = scio.loadmat('*/miniGCN/Test_X.mat')
Train_L = scio.loadmat('*/miniGCN/Train_L.mat')
Val_L = scio.loadmat('*/miniGCN/Val_L.mat')
Test_L = h5py.File('*/miniGCN/Test_L.mat')
Train_W = scio.loadmat('*/W/W_TR.mat')
Val_W = scio.loadmat('*/W/W_VAL.mat')

Train_X = Train_X['Train_X']
Val_X = Val_X['Val_X']
Test_X = Test_X['Test_X']
TrLabel = TrLabel['TrLabel']
ValLabel = ValLabel['ValLabel']
Train_W = Train_W['W_TR']
Val_W = Val_W['W_VAL']

Train_L = Train_L['Train_L']
Val_L = Val_L['VAL_L']
Test_L = np.array(Test_L['Test_L'], dtype=np.float16)

TrLabel = convert_to_one_hot(TrLabel, 2)
ValLabel = convert_to_one_hot(ValLabel, 2)
ValLabel = ValLabel.T
TrLabel = TrLabel.T
Train_W = Train_W.T
Val_W = Val_W.T


parameters, costs, costs_dev, train_acc, val_acc, features_out = train_mynetwork(Train_X, Val_X, Test_X, TrLabel,ValLabel, Train_L, Val_L, Test_L,Train_X, Val_X, Test_X, Train_W, Val_W)

# plot the cost
plt.plot(np.squeeze(costs), label='Training set')
plt.plot(np.squeeze(costs_dev), label='Validation set')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(loc='best')
plt.show()
#
# # plot the accuracy
plt.plot(np.squeeze(train_acc), label='Train set')
plt.plot(np.squeeze(val_acc), label='Validation set')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(loc='best')
plt.show()
