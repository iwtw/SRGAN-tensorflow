import tensorflow as tf
from utils import *

class Discriminator:
    def __init__(self,input,name='disc'):
        with tf.variable_scope(name):
            self.conv1=conv_layer(input,[3,3,3,64],1)
            self.lrelu1=leaky_relu(self.conv1)
            ochannels=[64,128,128,256,256,512,512]
            stride=[2,1]
            block=[self.lrelu1]
            for i in xrange(7):
                block.append(self.get_block(block[-1],ochannels[i],stride[i%2]))
            self.dense1=tf.layers.dense(block[-1],1024,
                                   kernel_initializer=tf.truncated_normal_initializer()
                                   )
            self.lrelu2=leaky_relu(self.dense1)
            self.dense2=tf.layers.dense(self.lrelu2,1,
                                   kernel_initializer=tf.truncated_normal_initializer(),
                                   activation=tf.sigmoid)
    def get_block(self,X,ochannels,stride,name='block'):
        with tf.variable_scope(name):
            X=conv_layer(X,[3,3,X.shape.as_list()[3],ochannels],stride)
            X=leaky_relu(X)
            X=batch_norm(X)
            return X
