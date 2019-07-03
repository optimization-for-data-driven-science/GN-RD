import matplotlib.pyplot as plt
import helper
import tensorflow as tf
import numpy as np
from PIL import Image
import scipy.io as sio
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)





sample_dir = 'logs/mnist14'
sample_dir_2 = './icp_format_results14/'
sample_dir_sample = 'logs/sample'

z_dim = 100
b_size = 100
max_iter = 100000
step_size = 1e-3
iteration = 500
hyper_parm_filter = {
'filter_size' : 5, ##### Best is 16
'fst_lyr_num_fltrs' : 64,
'scnd_lyr_num_fltrs' : 64,
}
rr = hyper_parm_filter['scnd_lyr_num_fltrs']



helper.folder_creator(sample_dir)
helper.folder_creator(sample_dir_2)
helper.folder_creator(sample_dir_sample)




W_conv1_temp, b_conv1_temp, W_conv2_temp, b_conv2_temp = helper.weight_bias(hyper_parm_filter)
W_conv1 = tf.get_variable(dtype = tf.float32, name = 'W_conv1',
initializer=W_conv1_temp)
b_conv1 = tf.get_variable(dtype = tf.float32,  name = 'b_conv1',
initializer=b_conv1_temp)
W_conv2 = tf.get_variable(dtype = tf.float32, name = 'W_conv2',
initializer=W_conv2_temp)
b_conv2 = tf.get_variable(dtype = tf.float32,  name = 'b_conv2',
initializer=b_conv2_temp)



D1 = {'W_conv1':W_conv1,'b_conv1':b_conv1, 'W_conv2':W_conv2,'b_conv2':b_conv2}
init_new_vars_op = tf.initialize_variables([W_conv1, b_conv1, W_conv2, b_conv2])

x1 = tf.placeholder(tf.float32, [None, 784], name = 'x1')
z1 = tf.placeholder(tf.float32, [None, z_dim], name = 'z1')

gz1 = helper.Generator(z1)


x1_real_conv = helper.random_discriminator(x1,rr,D1 )


gz1_real_conv = helper.random_discriminator(gz1,rr, D1)




loss_1 = helper.dist(x1_real_conv , gz1_real_conv, b_size)
loss_2 = helper.dist( gz1_real_conv, x1_real_conv, b_size)
g_loss =  tf.cond(tf.less(loss_1, loss_2), lambda: loss_2, lambda: loss_1)

gen_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "Generator")


step = tf.Variable(0, trainable=False)
rate = tf.train.exponential_decay(step_size, step, 3000, 0.96)


optimizer = tf.train.AdamOptimizer(learning_rate=rate, beta1=0.5, beta2=0.9, name = 'Adam1')
grads_and_vars = optimizer.compute_gradients(g_loss,  var_list=gen_params )
vars = [x[1] for x in grads_and_vars]
gradients = [x[0] for x in grads_and_vars]
gen_train_op = optimizer.apply_gradients(zip(gradients,vars), global_step = step )


#
def Create_feed_dict():
    filter_size = hyper_parm_filter['filter_size']
    fst_lyr_num_fltrs = hyper_parm_filter['fst_lyr_num_fltrs']
    scnd_lyr_num_fltrs = hyper_parm_filter['scnd_lyr_num_fltrs']
    feed = {}
    #
    feed[b_conv1_temp] = helper.return_const([fst_lyr_num_fltrs], c = 0.0)
    #
    feed[b_conv2_temp] = helper.return_const([scnd_lyr_num_fltrs], c = 0.0 )
    feed[W_conv1_temp] = helper.return_normal([filter_size, filter_size, 1, fst_lyr_num_fltrs])
    feed[W_conv2_temp] = helper.return_normal([filter_size, filter_size, fst_lyr_num_fltrs, scnd_lyr_num_fltrs])

    return feed

def Generate_samples_and_save(fixed_noise, iter = 0, counter = 1):
    fake_samples = sess.run(gz1, feed_dict={z1:fixed_noise})

    adict = {}
    adict['images'] = fake_samples
    c = counter
    sio.savemat(sample_dir_2+'{}.mat'.format(str(c).zfill(3)), adict)
    fake_samples = helper.data2img(fake_samples)
    sample_plot = fake_samples[np.random.randint(0, 10000, 128),:]


    fake_samples = helper.grid_transform(fake_samples, [28, 28, 1])
    fake_samples = np.squeeze(fake_samples)
    fake_samples = (255.99*fake_samples).astype('uint8')
    plt.imsave(sample_dir+'/samples_'+str(iter)+'.png', fake_samples)
    img = Image.open(sample_dir+'/samples_'+str(iter)+'.png').convert('LA')
    img.save(sample_dir+'/samples_'+str(iter)+'.png')
    sample_plot = helper.grid_transform2(sample_plot, [28, 28, 1])
    sample_plot = np.squeeze(sample_plot)
    sample_plot = (255.99 * sample_plot).astype('uint8')
    plt.imsave(sample_dir_sample + '/samples_' + str(iter) + '.png', sample_plot)
    img = Image.open(sample_dir_sample + '/samples_' + str(iter) + '.png').convert('LA')
    img.save(sample_dir_sample + '/samples_' + str(iter) + '.png')

    return fake_samples



fixed_noise = np.random.normal(0.0, 1.0, [10000, z_dim])

with tf.Session() as sess:

    feed2 = Create_feed_dict()
    sess.run(tf.global_variables_initializer(),feed_dict = feed2)
    iter = 0
    Plot_counter = 1
    Generate_samples_and_save(fixed_noise, iter = iter, counter = 0 )
    while iter < (max_iter):
        if iter % 10 == 0:
            print(iter)

        feed = {}
        data1 = mnist.train.next_batch(b_size)[0]
        code1 = helper.return_unifrom([b_size , z_dim])
        feed[x1] = data1
        feed[z1] = code1
        _= sess.run(gen_train_op, feed_dict = feed)

        iter = iter + 1
        feed2 = Create_feed_dict()
        sess.run(init_new_vars_op, feed_dict = feed2)

        if iter % 500 == 0:
            Generate_samples_and_save(fixed_noise, iter = iter, counter = Plot_counter )
            Plot_counter = Plot_counter  + 1

