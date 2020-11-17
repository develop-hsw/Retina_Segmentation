import tensorflow as tf
from tflearn.layers.conv import global_avg_pool
from tensorflow.contrib.layers import batch_norm, flatten
from tensorflow.contrib.framework import arg_scope
import numpy as np
from tensorflow.contrib import slim
"""
segmentation 모델로는 res unet이 사용 되었다. 
추가 segmentation network를 찾아서 사용하는 것이 좋을 것으로 보인다. 
"""
class Network():
    def __init__(self, x, class_num, training = True):
        self.training = training
        self.reduction_ratio = 4
        self.dropout_rate = 0.2
        self.class_num = class_num
        self.model = self.Build_unet(x)

    def resblock(self, inputs, out_channel=32, name='resblock'):

        with tf.variable_scope(name):
            x = tf.layers.conv2d(inputs=inputs, filters=out_channel, kernel_size=[3, 3], strides=1, padding='SAME',
                                 name='conv1')
            x = tf.nn.leaky_relu(x)
            x = tf.layers.conv2d(inputs=x, filters=out_channel, kernel_size=[3, 3], strides=1, padding='SAME',
                                 name='conv2')

            return x + inputs

    def Build_unet(self, inputs, channel=32, num_blocks=4, name='generator', reuse=False):
        with tf.variable_scope(name, reuse=reuse):
            x0 = tf.layers.conv2d(inputs=inputs, filters=channel, kernel_size=[7, 7], strides=1, padding='SAME')
            x0 = tf.nn.leaky_relu(x0)

            x1 = tf.layers.conv2d(inputs=x0, filters=channel, kernel_size=[3, 3], strides=2, padding='SAME')
            x1 = tf.nn.leaky_relu(x1)
            x1 = tf.layers.conv2d(inputs=x1, filters=channel * 2, kernel_size=[3, 3], strides=1, padding='SAME')
            x1 = tf.nn.leaky_relu(x1)

            x2 = tf.layers.conv2d(inputs=x1, filters=channel * 2, kernel_size=[3, 3], strides=2, padding='SAME')
            x2 = tf.nn.leaky_relu(x2)
            x2 = tf.layers.conv2d(inputs=x2, filters=channel * 4, kernel_size=[3, 3], strides=1, padding='SAME')
            x2 = tf.nn.leaky_relu(x2)

            for idx in range(num_blocks):
                x2 = self.resblock(x2, out_channel=channel * 4, name='block_{}'.format(idx))

            x2 = tf.layers.conv2d(inputs=x2, filters=channel * 2, kernel_size=[3, 3], strides=1, padding='SAME')
            x2 = tf.nn.leaky_relu(x2)

            h1, w1 = tf.shape(x2)[1], tf.shape(x2)[2]
            x3 = tf.image.resize_bilinear(x2, (h1 * 2, w1 * 2))
            # x3 = slim.convolution2d(x3 + x1, channel * 2, [3, 3], activation_fn=None)
            x3 = tf.layers.conv2d(inputs=x3 + x1, filters=channel * 2, kernel_size=[3, 3], strides=1, padding='SAME')
            x3 = tf.nn.leaky_relu(x3)
            x3 = tf.layers.conv2d(inputs=x3, filters=channel, kernel_size=[3, 3], strides=1, padding='SAME')
            x3 = tf.nn.leaky_relu(x3)

            h2, w2 = tf.shape(x3)[1], tf.shape(x3)[2]
            x4 = tf.image.resize_bilinear(x3, (h2 * 2, w2 * 2))
            x4 = tf.layers.conv2d(inputs=x4 + x0, filters=channel, kernel_size=[3, 3], strides=1, padding='SAME')
            x4 = tf.nn.leaky_relu(x4)
            x4 = tf.layers.conv2d(inputs=x4, filters=self.class_num, kernel_size=[7, 7], strides=1, padding='SAME')

            x4 = tf.nn.sigmoid(x4) #문제되는지 확인
            # x4 = tf.clip_by_value(x4, -1, 1)
        return x4

    def conv_layer(self,input, filter, kernel, stride=1, padding = "SAME",layer_name="conv",activation=True):
        with tf.name_scope(layer_name):
            network = tf.layers.conv2d(inputs=input, filters=filter, kernel_size=kernel, strides=stride, padding=padding)
            if activation:
                network = self.Relu(network)
            return network

    def Global_Average_Pooling(self, x, stride=1):
        return global_avg_pool(x, name='Global_avg_pooling')
        # But maybe you need to install h5py and curses or not

    def Batch_Normalization(self, x, training, scope):
        with arg_scope([batch_norm],
                       scope=scope,
                       updates_collections=None,
                       decay=0.9,
                       center=True,
                       scale=True,
                       zero_debias_moving_mean=True):
            return tf.cond(training,
                           lambda: batch_norm(inputs=x, is_training=training, reuse=None),
                           lambda: batch_norm(inputs=x, is_training=training, reuse=True))

    def Drop_out(self, x, rate, training):
        return tf.layers.dropout(inputs=x, rate=rate, training=training)

    def Relu(self, x):
        return tf.nn.relu(x)

    def Average_pooling(self, x, pool_size=[2, 2], stride=2, padding='VALID'):
        return tf.layers.average_pooling2d(inputs=x, pool_size=pool_size, strides=stride, padding=padding)

    def Max_Pooling(self, x, pool_size=[3, 3], stride=2, padding='VALID'):
        return tf.layers.max_pooling2d(inputs=x, pool_size=pool_size, strides=stride, padding=padding)

    def Concatenation(self, layers):
        return tf.concat(layers, axis=3)

    def flatten(self,x):
        return flatten(x)

    def Fully_connected(self, x, units, layer_name='fully_connected'):
        with tf.name_scope(layer_name):
            return tf.layers.dense(inputs=x, use_bias=True, units=units)

    def Sigmoid(self,x):
        return tf.nn.sigmoid(x)

    def Linear(self, x, layer_name = 'linear'):
        return tf.layers.dense(inputs=x, units=self.class_num, name=layer_name)
