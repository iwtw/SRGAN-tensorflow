import tensorflow as tf
import vgg19
import srResNet
import discriminator
import time
import random
import sys
import os 
import output as output
from utils import *
from skimage import io
import numpy as np

v1=3
v2=5
learn_rate1=0.1**v1
learn_rate2=0.1**v2
batch_size=32
H = 28 
W = 24
k=5
filenames='data.txt.aa'
if( len ( sys.argv ) == 1 ):
    name =""
else:
    name = sys.argv[1]
srResNet_path='save/srResNet'+name+'/'+"srResNet"
log_steps=100
num_epoch1=10
num_epoch2=20
save_path='save/srGANno'+name
save_file=save_path+'/srGANno'
output_path='./training_no_preoutput/'+name
if not os.path.exists(save_path):
    os.mkdir(save_path)
if not os.path.exists(output_path):
    os.mkdir(output_path)
def read(filenames):
    file_names=open(filenames,'r').read().split('\n')
    file_names.pop( len(file_names) -1 )
    steps_per_epoch = len(file_names) / batch_size
    random.shuffle(file_names)
    filename_queue=tf.train.string_input_producer(file_names)
    reader=tf.WholeFileReader()
    _,value=reader.read(filename_queue)
    image=tf.image.decode_jpeg(value)
    cropped=tf.random_crop(image,[ H *4, W*4,3])
    random_flipped=tf.image.random_flip_left_right(cropped)
    minibatch=tf.cast(tf.train.batch([random_flipped],batch_size,capacity=300),tf.float32)/255.0
    resized=tf.image.resize_bicubic(minibatch,[ H , W ])
    return steps_per_epoch , minibatch,resized

 
with tf.device('/cpu:0'):
    steps_per_epoch,minibatch,resized=read(filenames)
resnet=srResNet.srResNet(resized*2.0-1)
result=resnet.conv5
gen_var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)


dbatch=tf.concat([tf.cast(minibatch,tf.float32),result],0)
bicubic = tf.clip_by_value( tf.image.resize_bicubic( resized , [H*4,W*4] ) , 0 ,1  ) 
out = [ tf.cast( minibatch , tf.float32 ) ,result , bicubic ]

vgg=vgg19.Vgg19()
vgg.build(dbatch)
fmap=tf.split(vgg.conv2_2,2)
content_loss=tf.losses.mean_squared_error(fmap[0],fmap[1])

disc=discriminator.Discriminator(dbatch)
D_x,D_G_z=tf.split(disc.dense2,2)   

adv_loss=tf.reduce_mean(  ( 1.0 - D_G_z)**2  )
#gen_loss= 1e-3 * adv_loss + content_loss
gen_loss = content_loss
disc_loss=tf.reduce_mean(  (D_x)**2 + (1.0-D_G_z)**2  )

disc_var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
for x in gen_var_list:
    disc_var_list.remove(x)

global_step=tf.Variable(0,trainable=0,name='global_step')
gen_train_step1=tf.train.AdamOptimizer(learn_rate1).minimize(gen_loss,global_step,gen_var_list)
gen_train_step2=tf.train.AdamOptimizer(learn_rate2).minimize(gen_loss,global_step)
disc_train_step1=tf.train.AdamOptimizer(learn_rate1).minimize(disc_loss,global_step,disc_var_list)
disc_train_step2=tf.train.AdamOptimizer(learn_rate2).minimize(disc_loss,global_step)

config = tf.ConfigProto(allow_soft_placement=True , log_device_placement=False )
config.gpu_options.allow_growth=True
with tf.Session(config=config) as sess:
    if not os.path.exists(save_file+'.meta'):
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        saver=tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES))
        saver.save(sess,save_file)

    saver=tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES))
    saver.restore(sess,save_file)

    def save():
        saver.save(sess,save_file)
    sess.run(tf.local_variables_initializer())
    step=global_step.eval
    tf.train.start_queue_runners()

    endpoint1 = steps_per_epoch * num_epoch1
    endpoint2 = steps_per_epoch * num_epoch2
    
    #b,t,de,dgz,dx = sess.run([minibatch , result , disc.dense2,D_G_z,D_x ] )
    
    #print( np.min(b) , np.max(b) )
    #print( np.min(de) , np.max(de) )
    #print( np.min(dx) , np.max(dx) )
    #print( np.min(dgz) , np.max(dgz) )
    #print( sess.run([content_loss , adv_loss , disc_loss , gen_loss ]) )

    def train(endpoint,gen_step,disc_step):
        try:
            while step()<=endpoint:
                epoch = step() / steps_per_epoch
                for i in xrange(k):
                    sess.run(disc_step)
                sess.run(gen_step)
                if(step()%log_steps==0):
                    t = sess.run(result)
                    #print( np.min(t) , np.max(t) )
                    d_batch=dbatch.eval()
                    mse,psnr=batch_mse_psnr(d_batch)
                    ssim=batch_ssim(d_batch)
                    s=time.strftime('%Y-%m-%d %H:%M:%S:',time.localtime(time.time()))+"epoch="+str(step() / steps_per_epoch + 1 )+'step='+str(step())+' mse='+str(mse)+' psnr='+str(psnr)+' ssim='+str(ssim)+' gen_loss='+str(gen_loss.eval())+"_content_loss="+str(content_loss.eval())+"_adv_loss="+str(1e-3*adv_loss.eval())+' disc_loss='+str(disc_loss.eval())+"result:["+str(np.min(t))+","+str(np.max(t))+"]"
                    #print(s)
                    f=open('info.train_'+name,'a')
                    f.write(s+'\n')
                    f.close()
                    save()
                #if(True):
                if(step()%steps_per_epoch==0):
                    od = sess.run(out)
                #    for i in xrange(3):
                #        print(np.min(od[i]) , np.max(od[i]))
                    for i in xrange(od[0].shape[0]):
                        io.imsave(output_path+"/Epoch:"+str(epoch)+"No:"+str(i)+"_real.jpg",od[0][i])
                        io.imsave(output_path+"/Epoch:"+str(epoch)+"No:"+str(i)+"_generated.jpg",np.clip( od[1][i] ,0.0,1.0 ) )
                        io.imsave(output_path+"/Epoch:"+str(epoch)+"No:"+str(i)+"_bicubic.jpg",od[2][i])
        except tf.errors.OutOfRangeError:
            print('[INFO] train finished')
            save()
        except KeyboardInterrupt:
            print('[INFO] KeyboardInterrupt')
            save()
            exit()
            print('[INFO] checkpoint save done')
    train(endpoint1,gen_train_step1,disc_train_step1)
    train(endpoint2,gen_train_step2,disc_train_step2)
    print('trainning finished')
