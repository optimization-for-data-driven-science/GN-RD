import numpy as np
import tensorflow as tf
import os
import shutil

def folder_creator(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)

    for the_file in os.listdir(folder):
        file_path = os.path.join(folder, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(e)
def leaky_relu(x, alpha=0.2):
    return tf.maximum(x * alpha, x)

def conv2d(x, W):

    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')


def weight_placeholder(shape, name=None):
    return tf.placeholder(tf.float32, shape, name = name)

def bias_placeholder(shape, name = None):
    return tf.placeholder(tf.float32, shape, name = name)

def weight_bias(hyper):

    global filter_size, fst_lyr_num_fltrs, scnd_lyr_num_fltrs, third_lyr_num_fltrs, forth_lyr_num_fltrs

    filter_size = hyper['filter_size']
    fst_lyr_num_fltrs = hyper['fst_lyr_num_fltrs']
    scnd_lyr_num_fltrs = hyper['scnd_lyr_num_fltrs']

    W_conv1 = weight_placeholder([filter_size, filter_size, 1, fst_lyr_num_fltrs], name = 'W_conv1' )
    b_conv1 = bias_placeholder([fst_lyr_num_fltrs], name = 'b_conv1')

    W_conv2 = weight_placeholder([filter_size, filter_size, fst_lyr_num_fltrs, scnd_lyr_num_fltrs], name = 'W_conv2')
    b_conv2 = bias_placeholder([scnd_lyr_num_fltrs], name = 'b_conv2')

    return W_conv1, b_conv1, W_conv2, b_conv2


def Generator(z, reuse = False):



    with tf.variable_scope("Generator", reuse=reuse):
        bs = tf.shape(z)[0]
        fc1 = tf.layers.dense(z, 1024)
        fc1 = leaky_relu(fc1)
        fc2 = tf.layers.dense(fc1, 7 * 7 * 128)
        fc2 = tf.reshape(fc2, tf.stack([bs, 7, 7, 128]))
        fc2 = leaky_relu(fc2)
        conv1 = tf.contrib.layers.conv2d_transpose(fc2, 64, [4, 4], [2, 2])
        conv1 = leaky_relu(conv1)
        conv2 = tf.contrib.layers.conv2d_transpose(conv1, 1, [4, 4], [2, 2], activation_fn=tf.sigmoid)
        conv2 = tf.reshape(conv2, tf.stack([bs, 784]))
    return conv2



def random_discriminator(u, rr, D1):


    u = tf.reshape(u, [-1,28,28,1])
    u = leaky_relu(conv2d(u, D1['W_conv1']) + D1['b_conv1'])
    u = max_pool_2x2(u)

    u = leaky_relu(conv2d(u, D1['W_conv2']) + D1['b_conv2'])
    u = max_pool_2x2(u)
    u = tf.reshape(u, [-1, 7*7*rr]) #tf.contrib.layers.flatten(u)#


    return u


def dist(a, b, bs):
    c = 0.0 * tf.norm(a[1,] - b[1,])
    for j in range(bs):
       vec = a[j,]
       m = bs
       matrix = tf.ones([m, 1]) * vec
       temp = tf.norm(matrix - b, axis = 1)

       c = c + tf.reduce_min(temp)

    return c


def data2img(data):
    shape = [28, 28, 1]
    return np.reshape(data, [data.shape[0]] + shape)

def grid_transform2(x, size):
    a = 8
    b = 16
    h, w, c = size[0], size[1], size[2]
    x = np.reshape(x, [a, b, h, w, c])
    x = np.transpose(x, [0, 2, 1, 3, 4])
    x = np.reshape(x, [a * h, b * w, c])
    if x.shape[2] == 1:
        x = np.squeeze(x, axis=2)
    return x

def grid_transform(x, size):

    a = 10
    b = int(x.shape[0]/a)
    h, w, c = size[0], size[1], size[2]
    x = np.reshape(x, [a, b, h, w, c])
    x = np.transpose(x, [0, 2, 1, 3, 4])
    x = np.reshape(x, [a * h, b * w, c])
    if x.shape[2] == 1:
        x = np.squeeze(x, axis=2)
    return x

def return_normal(shape):
    return np.float32(np.random.normal(loc=0.0, scale=0.1, size=shape))

def return_const(shape, c = .1):
    return np.float32(np.full(shape = shape, fill_value = c))

def return_unifrom(shape):
    return np.float32(np.random.uniform(-1.0, 1.0, shape))

