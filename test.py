import tensorflow as tf
import numpy as np
import vgg19
import srResNet
import discriminator
import time
import random
import os 
from skimage import io
from utils import *
import sys
import matplotlib.image as mpimg

filenames='./testset_path.txt'
H=28
W=24
variable_path='./save/srGAN/srgan'
output_path="./outputdata/test/"
with tf.device('/cpu:0'):
    file_names=open(filenames,'r').read().split('\n')
    file_names.pop( len(file_names) -1 )

image = tf.placeholder(tf.float32 , shape=[1,H*4 , W*4 , 3 ])
#cropped = tf.random_crop(img,[ H *4, W*4,3])
#random_flipped=tf.image.random_flip_left_right(cropped)
rescaled=tf.image.resize_images( image, [H , W] , tf.image.ResizeMethod.BICUBIC)
#bicubic=tf.image.resize_images( rescaled , [H*4, W*4], tf.image.ResizeMethod.BICUBIC )
resnet=srResNet.srResNet(rescaled*(1.0/127.5) - 1.0)
result = ( (resnet.conv5)+1) * 127.5 
result_arr_rescaled = result * (1.0/255.0)
out = tf.clip_by_value( result_arr_rescaled , 0 , 1   )

config = tf.ConfigProto(allow_soft_placement=True , log_device_placement=False)
config.gpu_options.allow_growth=True
with tf.Session(config=config) as sess:
    loader = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES))
    loader.restore(sess , variable_path)
    for i in xrange(994):
        img = mpimg.imread(file_names[i])
        print(img.shape)
        outd=sess.run(out,feed_dict={image:[img]})
        io.imsave(output_path+file_names[i][40:],outd[0])


#outputdata(-1,50,"./data.txt.aa","./save/srGAN/srgan","./outputdata/")

