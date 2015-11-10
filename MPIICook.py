import numpy as np
import random
import cv2
import os
import itertools
from keras.utils import np_utils
from keras.utils.generic_utils import Progbar
from MPcallback import MPlogcallback
from keras.utils import np_utils
from makenetwork import makenetwork, makegraph
import readjson
import ujson
import keras.callbacks
import pdb
import os.path
import glob
import time
import sys
''' THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32,optimizer=fast_compile python MPIICook.py '''

isGraph = True
showAP = True
respath = 'merge2'
try:
    os.stat(respath)
except:    
    os.mkdir(respath)


nb_epoch = 100
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
        
np.random.shuffle(indtrain)
# pdb.set_trace()

print('make network and compile')
if isGraph:
    model = makegraph()
else:
    model = makenetwork()

found = glob.glob(respath+'/model*.hdf5')
mval = -1
if len(found) > 0:
    numlist = []
    for s in found:
        spl = s.split('.')
        num = spl[0]
        num = num[5:]
        numlist.append(int(num))
    mval = max(numlist)
    modpath = respath+'/model'+str(mval)+'.hdf5'
    print('load model from ' + modpath)
    model.load_weights(modpath)

if isGraph:
    model.compile(optimizer='rmsprop',loss={'actout':'binary_crossentropy', 'objout':'categorical_crossentropy'} )
else:
    model.compile(loss='binary_crossentropy', optimizer='rmsprop', class_mode="categorical")

# print('make checkpoint')
# checkpoint = keras.callbacks.ModelCheckpoint(filepath='model.hdf5', verbose=1, save_best_only=True)

MPLC = MPlogcallback()

jsonpath = '/home/rcvbong/jsonsMP'

if isGraph:
    tloss = []
    vloss = []
    tactap = []
    tobjap = []
    vactap = []
    vobjap = []
    # pdb.set_trace()
    if mval >= 0:
        f = open(respath+'/graphlossacc.txt','r')
        lines = f.read().splitlines()
        tlossstr = lines[0].split(',')
        vlossstr = lines[1].split(',')
        tactapstr = lines[2].split(',')
        tobjapstr = lines[3].split(',')
        vactapstr = lines[4].split(',')
        vobjapstr = lines[5].split(',')
        
        tlossstr = tlossstr[:len(tlossstr)-1]
        vlossstr = vlossstr[:len(vlossstr)-1]
        tactapstr = tactapstr[:len(tactapstr)-1]
        tobjapstr = tobjapstr[:len(tobjapstr)-1]
        vactapstr = vactapstr[:len(vactapstr)-1]
        vobjapstr = vobjapstr[:len(vobjapstr)-1]
        
        tloss = [float(x) for x in tlossstr]
        vloss = [float(x) for x in vlossstr]
        tactap = [float(x) for x in tactapstr]
        tobjap = [float(x) for x in tobjapstr]
        vactap = [float(x) for x in vactapstr]
        vobjap = [float(x) for x in vobjapstr]
        f.close()
else:
    tloss = []
    tacc = []
    vloss = []
    vacc = []
    tactap = []
    vactap = []
    # pdb.set_trace()
    if mval >= 0:
        f = open(respath+'/lossacc.txt','r')
        lines = f.read().splitlines()
        tlossstr = lines[0].split(',')
        taccstr = lines[1].split(',')
        vlossstr = lines[2].split(',')
        vaccstr = lines[3].split(',')
        tactapstr = lines[4].split(',')
        vactapstr = lines[5].split(',')
        
        tlossstr = tlossstr[:len(tlossstr)-1]
        taccstr = taccstr[:len(taccstr)-1]
        vlossstr = vlossstr[:len(vlossstr)-1]
        vaccstr = vaccstr[:len(vaccstr)-1]
        tactapstr = tactapstr[:len(tactapstr)-1]
        vactapstr = vactapstr[:len(vactapstr)-1]
        
        tloss = [float(x) for x in tlossstr]
        tacc = [float(x) for x in taccstr]
        vloss = [float(x) for x in vlossstr]
        vacc = [float(x) for x in vaccstr]
        tactap = [float(x) for x in tactapstr]
        vactap = [float(x) for x in vactapstr]
        f.close()
        
for iteration in range(mval+1,nb_epoch):
    # datapath='/media/disk1/bgsim/Dataset/UCF-101'
    # trainlist,testlist = makeDB(datapath=datapath,divideself=False)
    np.random.shuffle(indtrain)
    print('\nshuffle train index')
    print('\n'+str(iteration)+'th epoch '+'-'*50)
    
    seen = 0
    totloss = 0
    totacc = 0
    apseen = 0
    totobjAP = 0
    totactAP = 0
    progbar = Progbar(target=len(indtrain))
    for i in range(len(indtrain)):
        # pdb.set_trace()
        # if not os.path.isfile(os.path.join(jsonpath,str(indtrain[i])+'.json')):
            # print('no file exist')
            # continue
        start = time.time()
        # print('get DB from json files')
        X_train, act_label, obj_label = readjson.getDB(ind=indtrain[i])
        # y_train = readjson.to_categorical_dual(y_train,10)
        endread = time.time()
        
        if isGraph:
            model.fit({'input1':X_train,'actout':act_label,'objout':obj_label},batch_size=128,\
                        nb_epoch=1,shuffle=False,verbose=0,callbacks=[MPLC])
        else:    
        # print('model fit')
            model.fit(X_train,act_label,batch_size=128,nb_epoch=1,show_accuracy=True,shuffle=False,\
                      verbose=0,callbacks=[MPLC])

        endfit = time.time()

        # pdb.set_trace()
        
        progbar.update(i)

        if showAP:
            x_each = X_train.transpose(1,0,2)
            if isGraph:
                reterr = model.predict({'input1':x_each},verbose=0)
                predobj = reterr['objout']
                predact = reterr['actout']
                batchobjAP = 0
                batchactAP = 0
                batchacc = 0
                for batidx in range(len(predobj)):    
                    sortedobjerr = [si[0] for si in sorted(enumerate(predobj[batidx]),reverse=True,key=lambda xy:xy[1])]
                    itemobjidx = np.where(obj_label[batidx]==1)
                    # sortederr = [61,32,51,...] ( 0 ~ 154 )
                    # itemidx[0] = array([34,51,61]) ( 0 ~ 154 )
                    itemobjidx = itemobjidx[0]
                    
                    sortedacterr = [si[0] for si in sorted(enumerate(predact[batidx]),reverse=True,key=lambda xy:xy[1])]
                    itemactidx = np.where(act_label[batidx]==1)
                    # sortederr = [61,32,51,...] ( 0 ~ 154 )
                    # itemidx[0] = array([34,51,61]) ( 0 ~ 154 )
                    itemactidx = itemactidx[0][0]
                    
                    objAP = 0
                    
                    actpredidx = sortedacterr.index(itemactidx)
                    actAP = 1./(actpredidx+1)
                    batchactAP = batchactAP + actAP
                    
                    if actpredidx == 0:
                        batchacc = batchacc + 1
                    
                    soridx = []
                    for idx in range(len(itemobjidx)):
                        curidx = sortedobjerr.index(itemobjidx[idx])
                        curidx = curidx+1;
                        soridx.append(curidx)
                    soridx.sort()
                    for idx in range(len(soridx)):
                        objAP = objAP + float(idx+1)/soridx[idx]
                    batchobjAP = batchobjAP + float(objAP)/len(itemobjidx)
                
                totacc = totacc + batchacc
                
                totactAP = totactAP + batchactAP
                totobjAP = totobjAP + batchobjAP
                apseen = apseen + len(act_label)
                curactAP = float(totactAP)/apseen
                curobjAP = float(totobjAP)/apseen
                batactAP = float(batchactAP)/len(act_label)
                batobjAP = float(batchobjAP)/len(act_label)
                
                curacc = float(totacc)/apseen
                
                endap = time.time()
                info = ''
                info += ' batchact = %.2f' % batactAP
                info += ' batchobj = %.2f' % batobjAP
                info += ' curact = %.2f' % curactAP
                info += ' curobj = %.2f' % curobjAP
                info += ' acc = %.2f' % curacc
                info += ' time = %.2fs' % (endap-start)
                sys.stdout.write(info)
                sys.stdout.flush()                
            
                seen += (MPLC.seen)
                totloss += MPLC.totals.get('loss')
                
            else:
                reterr = model.predict(x_each,verbose=0)
                batchtotAP = 0
                for batidx in range(len(act_label)):    
                    sortederr = [si[0] for si in sorted(enumerate(reterr[batidx]),reverse=True,key=lambda xy:xy[1])]
                    itemidx = np.where(act_label[batidx]==1)
                    # sortederr = [61,32,51,...] ( 0 ~ 154 )
                    # itemidx[0] = array([34,51,61]) ( 0 ~ 154 )
                    itemidx = itemidx[0]
                    AP = 0
                    soridx = []
                    for idx in range(len(itemidx)):
                        curidx = sortederr.index(itemidx[idx])
                        curidx = curidx+1;
                        soridx.append(curidx)
                    soridx.sort()
                    for idx in range(len(soridx)):
                        AP = AP + float(idx+1)/soridx[idx]
                    batchtotAP = batchtotAP + float(AP)/len(itemidx)
                totactAP = totactAP + batchtotAP
                apseen = apseen + len(act_label)
                curAP = float(totactAP)/apseen
                batchAP = float(batchtotAP)/len(act_label)
                endap = time.time()
                info = ''
                info += ' batchAP = %.2f' % batchAP
                info += ' curAP = %.2f' % curAP
                info += ' read = %.2fs' % ((endread-start))
                info += ' fit = %.2fs' % ((endfit-endread))
                info += ' calcap = %.2fs' % ((endap-endfit))
                sys.stdout.write(info)
                sys.stdout.flush()                
            
                seen += (MPLC.seen)
                totloss += MPLC.totals.get('loss')
                totacc += MPLC.totals.get('acc')
        # vloss = MPLC.history.get('val_loss')
        # vacc = MPLC.history.get('val_acc')


    ### predict validations
    totvalloss = 0
    totvalscore = 0
    totvallen = 0
    valtotobjAP = 0
    valtotactAP = 0
    valapseen = 0
    print('\n'+str(iteration)+'th validation')
    progbar = Progbar(target=len(indval))
    for i in range(len(indval)):
        # if not os.path.isfile(os.path.join(jsonpath,str(indval[i])+'.json')):
        #     # print('no file exist')
        #     continue
        start = time.time()
        x_val, act_label, obj_label = readjson.getDB(ind=indval[i])
        x_each = x_val.transpose(1,0,2)
        endread = time.time()
        if not isGraph:
            score = model.evaluate(x_each,act_label,verbose=0,show_accuracy=True)
            totvalloss += score[0]
            totvalscore += score[1]
            totvallen += x_each.shape[0]
        else:
            score = model.evaluate({'input1':x_each,'actout':act_label,'objout':obj_label},verbose=0)
            # pdb.set_trace()
            totvalloss += score
            totvallen += x_each.shape[0]
            
        progbar.update(i)
        if showAP:
            if isGraph:
                reterr = model.predict({'input1':x_each},verbose=0)
                predobj = reterr['objout']
                predact = reterr['actout']
                batchobjAP = 0
                batchactAP = 0
                for batidx in range(len(predobj)):    
                    sortedobjerr = [si[0] for si in sorted(enumerate(predobj[batidx]),reverse=True,key=lambda xy:xy[1])]
                    itemobjidx = np.where(obj_label[batidx]==1)
                    # sortederr = [61,32,51,...] ( 0 ~ 154 )
                    # itemidx[0] = array([34,51,61]) ( 0 ~ 154 )
                    itemobjidx = itemobjidx[0]
                    
                    sortedacterr = [si[0] for si in sorted(enumerate(predact[batidx]),reverse=True,key=lambda xy:xy[1])]
                    itemactidx = np.where(act_label[batidx]==1)
                    # sortederr = [61,32,51,...] ( 0 ~ 154 )
                    # itemidx[0] = array([34,51,61]) ( 0 ~ 154 )
                    itemactidx = itemactidx[0][0]
                    
                    objAP = 0
                    
                    actpredidx = sortedacterr.index(itemactidx)
                    actAP = 1./(actpredidx+1)
                    batchactAP = batchactAP + actAP
                    
                    soridx = []
                    for idx in range(len(itemobjidx)):
                        curidx = sortedobjerr.index(itemobjidx[idx])
                        curidx = curidx+1;
                        soridx.append(curidx)
                    soridx.sort()
                    for idx in range(len(soridx)):
                        objAP = objAP + float(idx+1)/soridx[idx]
                    batchobjAP = batchobjAP + float(objAP)/len(itemobjidx)
                                        
                valtotactAP = valtotactAP + batchactAP
                valtotobjAP = valtotobjAP + batchobjAP
                valapseen = valapseen + len(act_label)
                curactAP = float(valtotactAP)/valapseen
                curobjAP = float(valtotobjAP)/valapseen
                batactAP = float(batchactAP)/len(act_label)
                batobjAP = float(batchobjAP)/len(act_label)
                endap = time.time()
                info = ''
                info += ' batchact = %.2f' % batactAP
                info += ' batchobj = %.2f' % batobjAP
                info += ' curact = %.2fs' % curactAP
                info += ' curobj = %.2fs' % curobjAP
                info += ' time = %.2fs' % (endap-start)
                sys.stdout.write(info)
                sys.stdout.flush()                
                       
            else:
                reterr = model.predict(x_each,verbose=0)
                endpred = time.time()
                batchtotAP = 0
                for batidx in range(len(act_label)):    
                    sortederr = [si[0] for si in sorted(enumerate(reterr[batidx]),reverse=True,key=lambda xy:xy[1])]
                    itemidx = np.where(act_label[batidx]==1)
                    # sortederr = [61,32,51,...] ( 0 ~ 154 )
                    # itemidx[0] = array([34,51,61]) ( 0 ~ 154 )
                    itemidx = itemidx[0]
                    AP = 0
                    soridx = []
                    for idx in range(len(itemidx)):
                        curidx = sortederr.index(itemidx[idx])
                        curidx = curidx+1;
                        soridx.append(curidx)
                    soridx.sort()
                    for idx in range(len(soridx)):
                        AP = AP + float(idx+1)/soridx[idx]
                    batchtotAP = batchtotAP + float(AP)/len(itemidx)
                valtotactAP = valtotactAP + batchtotAP
                valapseen = valapseen + len(act_label)
                curAP = float(valtotactAP)/valapseen
                batchAP = float(batchtotAP)/len(act_label)
                endap = time.time()
                info = ''
                info += ' batchAP = %.2f' % batchAP
                info += ' curAP = %.2f' % curAP
                info += ' read = %.2fs' % ((endread-start))
                info += ' pred = %.2fs' % ((endpred-endread))
                info += ' calcap = %.2fs' % ((endap-endpred))
                sys.stdout.write(info)
                sys.stdout.flush()                

    ### save model weights
    model.save_weights(respath+'/model'+str(iteration)+'.hdf5', overwrite=True)
    
    
    if isGraph:
        tloss.append(totloss/seen)
        vloss.append(totvalloss/len(indval))
        tactap.append(float(totactAP)/apseen)
        tobjap.append(float(totobjAP)/apseen)
        vactap.append(float(valtotactAP)/valapseen)
        vobjap.append(float(valtotobjAP)/valapseen)
    else:
        tloss.append(totloss/seen)
        vloss.append(totvalloss/len(indval))
        tacc.append(totacc/seen)
        vacc.append(totvalscore/len(indval))
        tactap.append(float(totactAP)/apseen)
        vactap.append(float(valtotactAP)/valapseen)
        
    if not isGraph:
        f = open(respath+'/lossacc.txt','w')
        for item in tloss:
            f.write("%f,"%item)
        f.write("\n")
        for item in tacc:
            f.write("%f,"%item)
        f.write("\n")
        for item in vloss:
            f.write("%f,"%item)
        f.write("\n")
        for item in vacc:
            f.write("%f,"%item)
        f.write("\n")
        for item in tactap:
            f.write("%f,"%item)
        f.write("\n")
        for item in vactap:
            f.write("%f,"%item)
        f.write("\n")        
        f.close()
    else:
        f = open(respath+'/graphlossacc.txt','w')
        for item in tloss:
            f.write("%f,"%item)
        f.write("\n")
        for item in vloss:
            f.write("%f,"%item)
        f.write("\n")
        for item in tactap:
            f.write("%f,"%item)
        f.write("\n")
        for item in tobjap:
            f.write("%f,"%item)
        f.write("\n")
        for item in vactap:
            f.write("%f,"%item)
        f.write("\n")
        for item in vobjap:
            f.write("%f,"%item)
        f.write("\n")             
        f.close()