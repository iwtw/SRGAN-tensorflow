import tensorflow as tf
import output as output
import srResNet
import time
import random
import sys
import os 
from utils import *
import vgg19

learn_rate=0.001
batch_size=32 #recommended
H = 28
W = 24
r = 4
filenames='data.txt.aa' #put images' paths to this file,one image path for each row,e.g. ./data/123.JPEG, or define another way of loading images in read()
log_steps=100 #interval to save the model parameters
num_epoch=15

if len(sys.argv)==1:
    name =""
else:
    name = sys.argv[1]
output_path="./pretraining_output/"+name


if not os.path.exists('save'):
    os.mkdir('save')
if not os.path.exists(output_path):
    os.mkdir(output_path)

save_path='save/srResNet'+name
save_file=save_path+'/srResNet'
if not os.path.exists(save_path):
    os.mkdir(save_path)

def read(filenames):
    file_names = open(filenames,'r').read()
    file_names = file_names.split('\n')
    file_names.pop(len(file_names) -1 )
    steps_per_epoch = len(file_names) / batch_size
#    for i,j in enumerate(file_names):
#        print(i,j)
    random.shuffle(file_names)
    filename_queue=tf.train.string_input_producer(file_names,capacity=3000)#shuffled input_producer by default
    reader=tf.WholeFileReader()
    _,value=reader.read(filename_queue)
    image=tf.image.decode_jpeg(value)
    cropped=tf.random_crop(image,[ H * r , W * r ,3])
    random_flipped=tf.image.random_flip_left_right(cropped)
    minibatch=tf.cast(tf.train.batch([random_flipped],batch_size,capacity=300),tf.float32)
    rescaled=tf.image.resize_bicubic(minibatch,[ H , W ])
    rescaled=rescaled*2/255-1
    return steps_per_epoch, minibatch,rescaled

with tf.device('/cpu:0'):
    steps_per_epoch,minibatch,rescaled=read(filenames)
resnet=srResNet.srResNet(rescaled)
result=(tf.tanh(resnet.conv5)+1)*255/2

dbatch=tf.concat([tf.cast(minibatch,tf.float32),result],0)
vgg=vgg19.Vgg19()
vgg.build(dbatch)
fmap=tf.split(vgg.conv2_2,2)
content_loss=tf.losses.mean_squared_error(fmap[0],fmap[1])
global_step=tf.Variable(0,name='global_step')
train_step=tf.train.AdamOptimizer(learn_rate).minimize(content_loss,global_step)

config = tf.ConfigProto(allow_soft_placement=True , log_device_placement=False )
config.gpu_options.allow_growth=True
with tf.Session(config=config) as sess:
    
    if not os.path.exists(save_file+'.meta'):
        sess.run(tf.local_variables_initializer())
        sess.run(tf.global_variables_initializer())
        saver=tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES))
        saver.save(sess,save_file)
    saver=tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES))
    saver.restore(sess,save_file)
    sess.run(tf.local_variables_initializer())
#    sess.run(tf.global_variables_initializer())
#    print('local init done')
    tf.train.start_queue_runners()
#    print('start queue done')
#    print("minibatch:",sess.run(minibatch))
#    print("result:",sess.run(result))
    def save():
        saver.save(sess,save_file)
    step=global_step.eval

    total_steps = steps_per_epoch * num_epoch
    
    try :
        while step()<=total_steps:
            sess.run(train_step)
            if(step()%log_steps==0 ):
                d_batch=dbatch.eval()
                mse,psnr=batch_mse_psnr(d_batch)
                ypsnr=batch_y_psnr(d_batch)
                ssim=batch_ssim(d_batch)
                s=time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))+' epoch:'+str(step()/steps_per_epoch+1)+':step '+str(step())+' mse:'+str(mse)+' psnr:'+str(psnr)+' ssim:'+str(ssim)+' y_psnr='+str(ypsnr)  
                print(s)
                f=open('info.pretrain_'+name,'a')
                f.write(s+'\n')
                f.close()
                save()
            if(step()%steps_per_epoch==0):
                output.outputdata(step()/steps_per_epoch , batch_size , filenames ,  save_file , output_path+'/')
    except tf.errors.OutOfRangeError:
        print('[INFO] train finished')
        save()
    except KeyboardInterrupt:
        print('[INFO] KeyboardInterrupt')
        save()
        print('[INFO] checkpoint save done')

print('done')
