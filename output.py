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
def outputdata( outputd , epoch , batch_size , filenames, variable_path,output_path , coord):

    
    loader = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES))
    loader.restore(sess , variable_path)
    sess.run(tf.local_variables_initializer())
    print("local init done ")
    output_thread = tf.train.start_queue_runners(sess=sess , coord = coord)
    print("queue runner start")
    print("before outd")
    outd=sess.run(out)
    print("forward done")
    #print( np.min(rescaledd ) , np.max(rescaledd) )
    return output_thread


#outputdata(-1,50,"./data.txt.aa","./save/srGAN/srgan","./outputdata/")

