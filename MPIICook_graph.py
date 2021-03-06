import numpy as np
import random
import cv2
import os
import itertools
from keras.utils import np_utils
from keras.utils.generic_utils import Progbar
from MPcallback import MPlogcallback
from keras.utils import np_utils
import makenetwork
import readjson
import ujson
import keras.callbacks
import pdb
import os.path
import glob
import time
import sys
import gc
from six.moves import cPickle
''' THEANO_FLAGS=mode=FAST_RUN,device=gpu0,floatX=float32,optimizer=fast_run,nvcc.fastmath=True python MPIICook_mer.py '''

nb_epoch = 100

print('make network and compile')
model = makenetwork.makemergeconcat2048()

rgbdir = '/media/disk1/bgsim/keras/MPIImemtest/RGBNormMul/'
ofdir = '/media/disk1/bgsim/keras/MPIImemtest/OFNormMul/'
names = ['1000.pkl.gz','2000.pkl.gz','3000.pkl.gz','4000.pkl.gz','5000.pkl.gz','6000.pkl.gz',\
         '7000.pkl.gz','8000.pkl.gz','9000.pkl.gz','10000.pkl.gz','10620.pkl.gz']                

respath = 'obj_concat_lstm2048'
logpath = respath+'/log.txt'

found = glob.glob(respath+'/model*.hdf5')
mval = -1
if len(found) > 0:
    numlist = []
    for s in found:
        spl = s.split('.')
        num = spl[0]
        splf = num.split('/')
        num = splf[1]
        num = num[5:]
        numlist.append(int(num))
    mval = max(numlist)
    modpath = respath+'/model'+str(mval)+'.hdf5'
    print('load model from ' + modpath)
    model.load_weights(modpath)

model.compile(optimizer='rmsprop',loss={'out':'categorical_crossentropy'} )

MPLC = MPlogcallback()

log = LOG(['tloss','vloss','tactap','vactap','tacc','vacc'])
if mval >= 0:
    log.loadlist(logpath)

namesord = range(len(names))
        
for iteration in range(mval+1,nb_epoch):
    # datapath='/media/disk1/bgsim/Dataset/UCF-101'
    # trainlist,testlist = makeDB(datapath=datapath,divideself=False)
    
    np.random.shuffle(namesord)
    
    seen = 0
    totloss = 0
    totacc = 0
    apseen = 0
    totobjAP = 0
    totactAP = 0
    timestepactAP = 0
    lastacc = 0
    print('\n'+str(iteration)+'th epoch '+'-'*50)
    
    progbar = Progbar(target=10620)
    
    passed = 0
    
    for nameidx in namesord:
    # for xxx in range(1):    
        
        # nameidx = namesord[0]
                
        f = open(rgbdir + names[nameidx])
        # f = open('1000.pkl.gz')
        curRGB = cPickle.load(f)
        f.close()
        
        f = open(ofdir + names[nameidx])
        # f = open('1000.pkl.gz')
        curOF = cPickle.load(f)
        f.close()
        
        # pdb.set_trace()
        
        RGBx_trainall = curRGB[0]
        act_labelall = curRGB[1]
        obj_labelall = curRGB[2]        
        # RGBact_labelall = curRGB[1]
        # RGBobj_labelall = curRGB[2]
    
        OFx_trainall = curOF[0]
        # OFact_labelall = curOF[1]
        # OFobj_labelall = curOF[2]
        
        trainlen = len(OFx_trainall)
        trainidxs = range(trainlen)
        np.random.shuffle(trainidxs)
        
        for tidx in range(len(trainidxs)):
            
            RGBX_train = RGBx_trainall[trainidxs[tidx]]
            OFX_train = OFx_trainall[trainidxs[tidx]]
            act_label = act_labelall[trainidxs[tidx]]
            obj_label = obj_labelall[trainidxs[tidx]]
                        
            # pdb.set_trace()
            X_train = np.append(RGBX_train,OFX_train,axis=2)
            cur_label = act_label
        
        # pdb.set_trace()
        # if not os.path.isfile(os.path.join(jsonpath,str(indtrain[i])+'.json')):
            # print('no file exist')
            # continue
            start = time.time()

            model.fit({'rgb':RGBX_train,'of':OFX_train,'out':cur_label},batch_size=128,\
                        nb_epoch=1,shuffle=False,verbose=0,callbacks=[MPLC])

            endfit = time.time()
    
        # pdb.set_trace()
        
            progbar.update(tidx+passed)
    
            reterr = model.predict({'rgb':RGBX_train,'of':OFX_train},verbose=0)
            batchtotAP = 0
            reterr = reterr['out']
            
            curAP, acc = getAP(reterr,cur_label)
            
            batchtotAP = batchtotAP + curAP
                   
            totactAP = totactAP + batchtotAP

            lastacc = lastacc + acc
            
            curAP = float(totactAP) / (tidx+passed+1)
            endap = time.time()
            info = ''
            info += ' acc = %.2f' % (float(lastacc)/(tidx+passed+1))
            info += ' batchAP = %.2f' % batchtotAP
            info += ' curAP = %.2f' % curAP
            # info += ' read = %.2fs' % ((endread-start))
            info += ' fit = %.2fs' % ((endfit-start))
            info += ' calcap = %.2fs' % ((endap-endfit))
            sys.stdout.write(info)
            sys.stdout.flush()
            
            seen += (MPLC.seen)
            totloss += MPLC.totals.get('loss')

        del curRGB, curOF, RGBx_trainall, act_labelall, obj_labelall, OFx_trainall
        gc.collect()
        passed = passed + trainlen

    ### save model weights
    model.save_weights(respath+'/model'+str(iteration)+'.hdf5', overwrite=True)

    ### predict validations
    totvalloss = 0
    totvalscore = 0
    totvallen = 0
    valtotobjAP = 0
    valtotactAP = 0
    valapseen = 0
    vallastacc = 0
    print('\n'+str(iteration)+'th validation')
    
    f = open(rgbdir + 'vals.pkl.gz')
    RGBvals = cPickle.load(f)
    f.close()
    
    f = open(ofdir + 'vals.pkl.gz')
    OFvals = cPickle.load(f)
    f.close()
    
    RGBx_valall = RGBvals[0]
    act_labelall = RGBvals[1]
    obj_labelall = RGBvals[2]
    
    OFx_valall = OFvals[0]
    
    vals = RGBvals
    
    progbar = Progbar(target=len(RGBvals[0]))
    for i in range(len(RGBvals[0])):
        # if not os.path.isfile(os.path.join(jsonpath,str(indval[i])+'.json')):
        #     # print('no file exist')
        #     continue
        start = time.time()
        RGBx_val = RGBx_valall[i]
        OFx_val = OFx_valall[i]
        act_label = act_labelall[i]
        obj_label = obj_labelall[i]
        
        x_val = np.append(RGBx_val,OFx_val,axis=2)
        
        cur_label = act_label
        
        endread = time.time()
       
        score = model.evaluate({'rgb':RGBx_val,'of':OFx_val,'out':cur_label},verbose=0)
        # pdb.set_trace()
        totvalloss += score
        totvallen += x_each.shape[0]            
            
        progbar.update(i)

        reterr = model.predict({'rgb':RGBx_val,'of':OFx_val},verbose=0)
        batchtotAP = 0
        reterr = reterr['out']
        curAP, acc = getAP(reterr,cur_label)
        batchtotAP = batchtotAP + curAP
               
        valtotactAP = valtotactAP + batchtotAP
        
        vallastacc = vallastacc + acc               
        curAP = float(valtotactAP) / (i+1)
        endap = time.time()
        info = ''
        info += ' acc = %.2f' % (float(vallastacc)/(i+1))
        info += ' batchAP = %.2f' % batchtotAP
        info += ' curAP = %.2f' % curAP
        # info += ' read = %.2fs' % ((endread-start))
        # info += ' fit = %.2fs' % ((endfit-start))
        info += ' time = %.2fs' % ((endap-start))
        sys.stdout.write(info)
        sys.stdout.flush()     
                
                
    # log = LOG(['tloss','vloss','tactap','vactap','tacc','vacc'])                
    log.appendlist([totloss/passed,totvalloss/len(vals[0]),float(totactAP)/passed,float(valtotactAP)/len(vals[0]),\
                    float(lastacc)/passed,float(vallastacc)/len(vals[0])])
    
    log.savelist(logpath)
    
    del RGBvals, OFvals, RGBx_valall, act_labelall, obj_labelall, OFx_valall, vals
    gc.collect()
