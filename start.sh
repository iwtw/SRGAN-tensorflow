#!/bin/bash
name=mse+1.0*vgg
nohup python -u pretrain.py $name  >> info.pretrain_$name &
