import tensorflow as tf
from tensorflow.contrib import layers
import math

arg_scope = tf.contrib.framework.arg_scope


def basic_block(l, nOutChannels, stride=1, name=None):

    assert len(nOutChannels) == 2

    with tf.variable_scope(name):
        nInChannels = l.get_shape()[1].value
        preact = layers.batch_norm(l, activation_fn=tf.nn.relu, scope='pre_bn')

        if nInChannels == nOutChannels[-1]:
            if stride == 1:
                shortcut = l
            else:
                shortcut = layers.max_pool2d(preact, 1, stride=stride, scope='shortcut')
        else:
            shortcut = layers.conv2d(preact, nOutChannels[1], 1, stride=stride, scope='shortcut')

        o = layers.conv2d(preact, nOutChannels[0], 3, stride=stride, scope='conv0')
        o = layers.batch_norm(o, activation_fn=tf.nn.relu, scope='bn0')
        o = layers.conv2d(o, nOutChannels[1], 3, stride=1, scope='conv1')
        o = o + shortcut
        return o


def bottleneck_block(l, nOutChannels, stride=1, name=None):

    assert len(nOutChannels) == 3

    with tf.variable_scope(name):
        nInChannels = l.get_shape()[1].value
        preact = layers.batch_norm(l, activation_fn=tf.nn.relu, scope='pre_bn')

        if nInChannels == nOutChannels[-1]:
            if stride == 1:
                shortcut = l
            else:
                shortcut = layers.max_pool2d(preact, 1, stride=stride, scope='shortcut')
        else:
            shortcut = layers.conv2d(preact, nOutChannels[2], 1, stride=stride, scope='shortcut')

        o = layers.conv2d(preact, nOutChannels[0], 1, stride=stride, scope='conv0')
        o = layers.batch_norm(o, activation_fn=tf.nn.relu, scope='bn0')
        o = layers.conv2d(o, nOutChannels[1], 3, stride=1, scope='conv1')
        o = layers.batch_norm(o, activation_fn=tf.nn.relu, scope='bn1')
        o = layers.conv2d(o, nOutChannels[2], 1, stride=1, scope='conv2')
        o = o + shortcut
        return o


def inference(images, n_class, output=None, is_bottleneck=False, is_train=False):

    depth = 1010
    n = (depth - 2) // 12
    # net_def = [(23, 58, 229, 23), ([24, 24, 24*3], [48, 48, 48*3], [96, 96, 96*3], [192, 192, 192*3])]
    n_stage = [16, 48, 96, 192, 384]
    net_def = [[n]*4,
               [(i, o//4, o) for i, o in zip(n_stage[:-1], n_stage[1:])]]

    with arg_scope([layers.conv2d, layers.max_pool2d,
                    layers.avg_pool2d, layers.batch_norm],
                    data_format='NCHW'):

        with arg_scope([layers.conv2d, layers.batch_norm, layers.fully_connected],
                activation_fn=None):

            with arg_scope([layers.batch_norm],
                    fused=True, scale=True, is_training=is_train):

                net = layers.conv2d(images, 32, 5, stride=1, scope='conv0')
                        
                for k, (n, nOutChannels) in enumerate(zip(*net_def)):
                    stride = 2
                    for i in range(0, n):
                        if len(nOutChannels) == 2:
                            block = basic_block
                        else:
                            block = bottleneck_block
                        net = block(net, nOutChannels, stride=stride, name='res%d_%d' % (k, i))
                        stride = 1

                net = layers.batch_norm(net, activation_fn=tf.nn.relu, scope='final_bn')
                # net = layers.avg_pool2d(net, net.get_shape().as_list()[-2:], 1, scope='global_avg')
                net = layers.max_pool2d(net, 2, 2, scope='final_pool')

                net = layers.flatten(net)
                fc5 = layers.fully_connected(net, 128, scope='fc5')

                logits = None
                if n_class:
                    logits = layers.fully_connected(fc5, n_class, scope='fc6')

    return (logits, fc5) if logits != None else fc5
