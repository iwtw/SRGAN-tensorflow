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

v1=3
v2=5
learn_rate1=0.1**v1
learn_rate2=0.1**v2
H = 28 
W = 24
def outputdata( epoch , batch_size , filenames, variable_path,output_path):
    with tf.device('/cpu:0'):
        file_names=open(filenames,'r').read().split('\n')
        file_names.pop( len(file_names) -1 )
        steps_per_epoch = len(file_names) / batch_size
        random.shuffle(file_names)
        filename_queue=tf.train.string_input_producer(file_names)
        reader=tf.WholeFileReader()
        _,value=reader.read(filename_queue)
    image=tf.image.decode_jpeg(value)
    cropped=tf.random_crop(image,[ H *4, W*4,3])
    #random_flipped=tf.image.random_flip_left_right(cropped)
    minibatch=tf.train.batch( [cropped] ,batch_size,capacity=300)
    rescaled=tf.image.resize_images(minibatch, [H , W] , tf.image.ResizeMethod.BICUBIC)
    bicubic=tf.image.resize_images( rescaled , [H*4, W*4], tf.image.ResizeMethod.BICUBIC )
    resnet=srResNet.srResNet(rescaled*(1.0/127.5) - 1.0)
    result = ( (resnet.conv5)+1) * 127.5 
    result_arr= tf.concat( [ tf.cast(minibatch , tf.float32) , result , bicubic ] , axis = 0 )
    result_arr_rescaled = result_arr * (1.0/255.0)
    outarr = tf.clip_by_value( result_arr_rescaled , 0 , 1   )
    out = tf.split(outarr,3,axis=0) 
    
    config = tf.ConfigProto(allow_soft_placement=True , log_device_placement=False)
    config.gpu_options.allow_growth=True
    with tf.Session(config=config) as sess:
        loader = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES))
        loader.restore(sess , variable_path)
        sess.run(tf.local_variables_initializer())
        tf.train.start_queue_runners()
        outd=sess.run(out)
        #print( np.min(rescaledd ) , np.max(rescaledd) )
        for i in xrange(3):
            print(np.min(outd[i]) , np.max(outd[i]))
        for i in xrange(outd[0].shape[0]):
            io.imsave(output_path+"Epoch:"+str(epoch)+"No:"+str(i)+"_real.jpg",outd[0][i])
            io.imsave(output_path+"Epoch:"+str(epoch)+"No:"+str(i)+"_generated.jpg",outd[1][i])
            io.imsave(output_path+"Epoch:"+str(epoch)+"No:"+str(i)+"_bicubic.jpg",outd[2][i])


#outputdata(-1,50,"./data.txt.aa","./save/srGAN/srgan","./outputdata/")

