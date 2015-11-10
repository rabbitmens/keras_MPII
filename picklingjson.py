import numpy as np
import random
import cv2
import os
import itertools
from keras.utils import np_utils
from keras.utils.generic_utils import Progbar
from keras.utils import np_utils
import readjson
import ujson
import keras.callbacks
import pdb
import os.path
import glob
import time
import sys
from six.moves import cPickle
from six.moves import range

''' THEANO_FLAGS=mode=FAST_RUN,device=gpu0,floatX=float32,optimizer=fast_compile python picklingjson.py '''

def normalized(x, axis=2, order=2, mul=100):
    l2 = np.atleast_1d(np.linalg.norm(x,order,axis))
    l2[l2==0]=1
    norm = x / np.expand_dims(l2,axis)
    
    norm = norm * mul
    return norm.astype('float16')

f = open('sets.json','r')
json = ujson.loads(f.read())
f.close()
shell = json.values()
data = shell[0]
setlist = data['sets']
indtrain = []
indval = []
indtest = []
for i,x in enumerate(setlist):
    if x == 1:
        indtrain.append(i+1)
    elif x == 2:
        indval.append(i+1)
    elif x == 3:
        indtest.append(i+1)
                
jsonpath = '/media/disk1/bgsim/Subactivity_Multistream/jsonsMPrgb'

savefold = 'RGBNormMul/'

trainall = []
actall = []
objall = []
tofile = []
print('train = '+ str(len(indtrain)))
totlen = len(indtrain)
for i in range(totlen):
    if i%50 == 0:
        print('cur ' +str(i))
    if i%1000 == 0:
        f = open(savefold+str(i)+'.pkl.gz','w')
        tofile.append(trainall)
        tofile.append(actall)
        tofile.append(objall)
        cPickle.dump(tofile,f)
        f.close()
        tofile = []
        trainall = []
        actall = []
        objall = []
    X_train, act_label, obj_label = readjson.getDB(jsonpath=jsonpath,ind=indtrain[i])
    # pdb.set_trace()
    X_train = normalized(X_train)
    trainall.append(X_train)
    # at here, all labels are same. should it be appended?
    actall.append(act_label)
    objall.append(obj_label)

# f = open('vals.pkl.gz','w')
f = open(savefold+str(totlen)+'.pkl.gz','w')
tofile.append(trainall)
tofile.append(actall)
tofile.append(objall)
cPickle.dump(tofile,f)
f.close()
tofile = []
trainall = []
actall = []
objall = []


trainall = []
actall = []
objall = []
tofile = []
print('val = ' + str(len(indval)))
totlen = len(indval)
for i in range(totlen):
    if i%50 == 0:
        print('cur ' +str(i))
    if i%1000 == 0:
        f = open(savefold+str(i)+'.pkl.gz','w')
        tofile.append(trainall)
        tofile.append(actall)
        tofile.append(objall)
        cPickle.dump(tofile,f)
        f.close()
        tofile = []
        trainall = []
        actall = []
        objall = []
    X_train, act_label, obj_label = readjson.getDB(jsonpath=jsonpath,ind=indval[i])
    X_train = normalized(X_train)
    trainall.append(X_train)
    # at here, all labels are same. should it be appended?
    actall.append(act_label)
    objall.append(obj_label)

# f = open('vals.pkl.gz','w')
f = open(savefold+'vals.pkl.gz','w')
tofile.append(trainall)
tofile.append(actall)
tofile.append(objall)
cPickle.dump(tofile,f)
f.close()
tofile = []
trainall = []
actall = []
objall = []



trainall = []
actall = []
objall = []
tofile = []
print('test = '+str(len(indtest)))
totlen = len(indtest)
for i in range(totlen):
    if i%50 == 0:
        print('cur ' +str(i))
    if i%1000 == 0:
        f = open(savefold+'test'+str(i)+'.pkl.gz','w')
        tofile.append(trainall)
        tofile.append(actall)
        tofile.append(objall)
        cPickle.dump(tofile,f)
        f.close()
        tofile = []
        trainall = []
        actall = []
        objall = []
    X_train, act_label, obj_label = readjson.getDB(jsonpath=jsonpath,ind=indtest[i])
    X_train = normalized(X_train)
    trainall.append(X_train)
    # at here, all labels are same. should it be appended?
    actall.append(act_label)
    objall.append(obj_label)

# f = open('vals.pkl.gz','w')
f = open(savefold+'test'+str(totlen)+'.pkl.gz','w')
tofile.append(trainall)
tofile.append(actall)
tofile.append(objall)
cPickle.dump(tofile,f)
f.close()
tofile = []
trainall = []
actall = []
objall = []

