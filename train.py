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
srResNet_path='./save/srResNet'+name+'/'+"srResNet"
log_steps=100
num_epoch1=10
num_epoch2=20
save_path='save/srGAN'+name
save_file=save_path+'/srGAN'
output_path='./training_output/'+name
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
    minibatch=tf.train.batch([random_flipped],batch_size,capacity=300)
    rescaled=tf.image.resize_bicubic(minibatch,[ H , W ])/127.5-1
    return steps_per_epoch , minibatch,rescaled

 
with tf.device('/cpu:0'):
    steps_per_epoch,minibatch,rescaled=read(filenames)
resnet=srResNet.srResNet(rescaled)
result=(resnet.conv5+1)*127.5
gen_var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)


dbatch=tf.concat([tf.cast(minibatch,tf.float32),result],0)
vgg=vgg19.Vgg19()
vgg.build(dbatch)
fmap=tf.split(vgg.conv2_2,2)
content_loss=tf.losses.mean_squared_error(fmap[0],fmap[1])

disc=discriminator.Discriminator(dbatch)
D_x,D_G_z=tf.split(tf.squeeze(disc.dense2),2)   

adv_loss=tf.reduce_mean( tf.log( 1.0 - D_G_z ) )
gen_loss= 1e-3 * adv_loss + content_loss
disc_loss=(tf.reduce_mean(tf.log(1.0-D_x)+tf.log(D_G_z)))

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
        loader = tf.train.Saver(var_list=gen_var_list)
        loader.restore(sess,srResNet_path)
        saver=tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES))
        saver.save(sess,save_file)
        print('saved')
    saver=tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES))
    saver.restore(sess,save_file)
    def save():
        saver.save(sess,save_file)
    sess.run(tf.local_variables_initializer())
    step=global_step.eval
    tf.train.start_queue_runners()
#    print("start queue done")
#    print(sess.run(gen_var_list))

    endpoint1 = steps_per_epoch * num_epoch1
    endpoint2 = steps_per_epoch * num_epoch2

    def train(endpoint,gen_step,disc_step):
        try:
            while step()<=endpoint:
                for i in xrange(k):
                    sess.run(disc_step)
                sess.run(gen_step)
                if(step()%log_steps==0):
                    d_batch=dbatch.eval()
                    mse,psnr=batch_mse_psnr(d_batch)
                    ssim=batch_ssim(d_batch)
                    s=time.strftime('%Y-%m-%d %H:%M:%S:',time.localtime(time.time()))+"epoch="+str(step() / steps_per_epoch + 1 )+'step='+str(step())+' mse='+str(mse)+' psnr='+str(psnr)+' ssim='+str(ssim)+' gen_loss='+str(gen_loss.eval())+' disc_loss='+str(disc_loss.eval())
                    print(s)
                    f=open('info.train_'+name,'a')
                    f.write(s+'\n')
                    f.close()
                    save()
                if(step()%steps_per_epoch==0):
                    output.outputdata(step()/steops_per_epoch , batch_size , filename , save_file , output_path+'/')
        except tf.errors.OutOfRangeError:
            print('[INFO] train finished')
            save()
        except KeyboardInterrupt:
            print('[INFO] KeyboardInterrupt')
            save()
            print('[INFO] checkpoint save done')
    train(endpoint1,gen_train_step1,disc_train_step1)
    train(endpoint2,gen_train_step2,disc_train_step2)
    print('trainning finished')
