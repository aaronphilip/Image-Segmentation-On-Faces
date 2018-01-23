# -*- coding: utf-8 -*-
"""
Created on Tue Jan  2 22:26:55 2018

@author: aaron
"""
from random import shuffle
import glob
import numpy as np
import tables
from PIL import Image

input_paths = ['Tensorflow-Segmentation-master/data28_28/inputs/*.jpg',
               'Tensorflow-Segmentation-master/data128_128/inputs/*.jpg',
               'Tensorflow-Segmentation-master/data250_250/inputs/*.jpg']

target_paths = ['Tensorflow-Segmentation-master/data28_28/targets_face_only/*.jpg',
               'Tensorflow-Segmentation-master/data128_128/targets_face_only/*.jpg',
               'Tensorflow-Segmentation-master/data250_250/targets_face_only/*.jpg']

input_addrs = []
target_addrs = []

for path in input_paths:
    input_addrs.extend(glob.glob(path))
    
for path in target_paths:
    target_addrs.extend(glob.glob(path))
    
c = list(zip(input_addrs,target_addrs))
shuffle(c)
input_addrs, target_addrs = zip(*c)

train_input = input_addrs[0:int(0.6*len(input_addrs))]
train_target = target_addrs[0:int(0.6*len(target_addrs))]

val_input = input_addrs[int(0.6*len(input_addrs)):int(0.8*len(input_addrs))]
val_target = target_addrs[int(0.6*len(target_addrs)):int(0.8*len(target_addrs))]

test_input = input_addrs[int(0.8*len(input_addrs)):]
test_target = target_addrs[int(0.8*len(target_addrs)):]

i_shape = (0,224,224, 3)
t_shape = (0,224,224)
dt = tables.Float32Atom()

f = tables.open_file('data.hdf5', mode='w')

train_i_arr = f.create_earray(f.root, 'train_i', dt, shape=i_shape)
val_i_arr = f.create_earray(f.root, 'val_i', dt, shape=i_shape)
test_i_arr = f.create_earray(f.root, 'test_i', dt, shape=i_shape)

train_t_arr = f.create_earray(f.root, 'train_t', dt, shape=t_shape)
val_t_arr = f.create_earray(f.root, 'val_t', dt, shape=t_shape)
test_t_arr = f.create_earray(f.root, 'test_t', dt, shape=t_shape)

for i in range(len(train_input)):
    i_addr = train_input[i]
    t_addr = train_target[i]
    
    i_img = Image.open(i_addr).convert('RGB')
    t_img = Image.open(t_addr).convert('L')
    
    i_img = i_img.resize((224,224))
    t_img = t_img.resize((224,224))
    
    i_img = np.array(i_img).astype(np.float32)
    t_img = (np.array(t_img) > 255. / 1.2).astype(np.float32)
    
    train_i_arr.append(i_img[None])
    train_t_arr.append(t_img[None])
    
for i in range(len(val_input)):
    i_addr = val_input[i]
    t_addr = val_target[i]
    
    i_img = Image.open(i_addr).convert('RGB')
    t_img = Image.open(t_addr).convert('L')
    
    i_img = i_img.resize((224,224))
    t_img = t_img.resize((224,224))
    
    i_img = np.array(i_img).astype(np.float32)
    t_img = (np.array(t_img) > 255. / 1.2).astype(np.float32)
    
    val_i_arr.append(i_img[None])
    val_t_arr.append(t_img[None])
    
for i in range(len(test_input)):
    i_addr = test_input[i]
    t_addr = test_target[i]
    
    i_img = Image.open(i_addr)
    t_img = Image.open(t_addr)
    
    i_img = i_img.resize((224,224)).convert('RGB')
    t_img = t_img.resize((224,224)).convert('L')
    
    i_img = np.array(i_img).astype(np.float32)
    t_img = (np.array(t_img) > 255. / 1.2).astype(np.float32)
    
    test_i_arr.append(i_img[None])
    test_t_arr.append(t_img[None])

f.close()
