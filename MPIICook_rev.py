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
''' THEANO_FLAGS=mode=FAST_RUN,device=gpu0,floatX=float32,optimizer=fast_compile python MPIICook_rev.py '''

ismerge = False
isGraph = False
showAP = True

nb_epoch = 100

print('make network and compile')
if isGraph:
    model = makenetwork.makegraph()
elif ismerge:
    model = makenetwork.makemergedrop2()
else:
    model = makenetwork.makeactnet()


respath = 'act_new_rgb'

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

if isGraph:
    model.compile(optimizer='rmsprop',loss={'actout':'binary_crossentropy', 'objout':'categorical_crossentropy'} )
elif ismerge:
    model.compile(optimizer='rmsprop',loss={'out':'binary_crossentropy'} )
else:
    model.compile(loss='binary_crossentropy', optimizer='rmsprop', class_mode="categorical")


MPLC = MPlogcallback()

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

elif ismerge:        

    tloss = []
    vloss = []
    tactap = []    
    vactap = []
    
    # pdb.set_trace()
    if mval >= 0:
        f = open(respath+'/mergelossacc.txt','r')
        lines = f.read().splitlines()
        tlossstr = lines[0].split(',')
        vlossstr = lines[1].split(',')
        tactapstr = lines[2].split(',')
        vactapstr = lines[3].split(',')
        
        tlossstr = tlossstr[:len(tlossstr)-1]
        vlossstr = vlossstr[:len(vlossstr)-1]
        tactapstr = tactapstr[:len(tactapstr)-1]
        vactapstr = vactapstr[:len(vactapstr)-1]
        
        tloss = [float(x) for x in tlossstr]
        vloss = [float(x) for x in vlossstr]
        tactap = [float(x) for x in tactapstr]
        vactap = [float(x) for x in vactapstr]
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
        
rgbdir = '/media/disk1/bgsim/keras/MPIImemtest/RGBNormtrain/'
ofdir = '/media/disk1/bgsim/keras/MPIImemtest/OFNormtrain/'
names = ['1000.pkl.gz','2000.pkl.gz','3000.pkl.gz','4000.pkl.gz','5000.pkl.gz','6000.pkl.gz',\
         '7000.pkl.gz','8000.pkl.gz','9000.pkl.gz','10000.pkl.gz','10620.pkl.gz']        
        
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
        
        trainlen = len(RGBx_trainall)
        trainidxs = range(trainlen)
        np.random.shuffle(trainidxs)
        
        for tidx in range(len(trainidxs)):
            
            RGBX_train = RGBx_trainall[trainidxs[tidx]]
            OFX_train = OFx_trainall[trainidxs[tidx]]
            act_label = act_labelall[trainidxs[tidx]]
            obj_label = obj_labelall[trainidxs[tidx]]
                        
            # pdb.set_trace()
            
            # X_train = np.append(RGBX_train,OFX_train,axis=2)    
            X_train = RGBX_train
            cur_label = act_label
        
        # pdb.set_trace()
        # if not os.path.isfile(os.path.join(jsonpath,str(indtrain[i])+'.json')):
            # print('no file exist')
            # continue
            start = time.time()
            
            if isGraph:
                model.fit({'input1':X_train,'actout':act_label,'objout':obj_label},batch_size=128,\
                            nb_epoch=1,shuffle=False,verbose=0,callbacks=[MPLC])
            elif ismerge:
                model.fit({'rgb':RGBX_train,'of':OFX_train,'out':cur_label},batch_size=128,\
                            nb_epoch=1,shuffle=False,verbose=0,callbacks=[MPLC])
            else:    
            # print('model fit')
                model.fit(X_train,cur_label,batch_size=128,nb_epoch=1,show_accuracy=True,shuffle=False,\
                          verbose=0,callbacks=[MPLC])
    
            endfit = time.time()
    
        # pdb.set_trace()
        
            progbar.update(tidx+passed)
    
            if showAP:
                
                if isGraph:
                    reterr = model.predict({'input1':X_train},verbose=0)
                    predobj = reterr['objout']
                    predact = reterr['actout']
                    batchobjAP = 0
                    batchactAP = 0
                      
                    sortedobjerr = [si[0] for si in sorted(enumerate(predobj[0]),reverse=True,key=lambda xy:xy[1])]
                    itemobjidx = np.where(obj_label[0]==1)
                    # sortederr = [61,32,51,...] ( 0 ~ 154 )
                    # itemidx[0] = array([34,51,61]) ( 0 ~ 154 )
                    itemobjidx = itemobjidx[0]
                    
                    sortedacterr = [si[0] for si in sorted(enumerate(predact[0]),reverse=True,key=lambda xy:xy[1])]
                    itemactidx = np.where(act_label[0]==1)
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
                                            
                    totactAP = totactAP + batchactAP
                    totobjAP = totobjAP + batchobjAP
                    apseen = apseen + len(act_label)
                    curactAP = float(totactAP)/apseen
                    curobjAP = float(totobjAP)/apseen
                    batactAP = float(batchactAP)/len(act_label)
                    batobjAP = float(batchobjAP)/len(act_label)
                    endap = time.time()
                    info = ''
                    info += ' batchact = %.2f' % batactAP
                    info += ' batchobj = %.2f' % batobjAP
                    info += ' curact = %.2f' % curactAP
                    info += ' curobj = %.2f' % curobjAP
                    info += ' time = %.2fs' % (endap-start)
                    sys.stdout.write(info)
                    sys.stdout.flush()                
                
                    seen += (MPLC.seen)
                    totloss += MPLC.totals.get('loss')
                    
                elif ismerge:
                    reterr = model.predict({'rgb':RGBX_train,'of':OFX_train},verbose=0)
                    batchtotAP = 0
                    reterr = reterr['out']
                    
                    sortederr = [si[0] for si in sorted(enumerate(reterr[0]),reverse=True,key=lambda xy:xy[1])]
                    itemidx = np.where(cur_label[0]==1)
                    # pdb.set_trace()
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
                    
                    curAP = float(totactAP) / (tidx+passed+1)
                    endap = time.time()
                    info = ''
                    info += ' batchAP = %.2f' % batchtotAP
                    info += ' curAP = %.2f' % curAP
                    # info += ' read = %.2fs' % ((endread-start))
                    info += ' fit = %.2fs' % ((endfit-start))
                    info += ' calcap = %.2fs' % ((endap-endfit))
                    sys.stdout.write(info)
                    sys.stdout.flush()
                    
                    seen += (MPLC.seen)
                    totloss += MPLC.totals.get('loss')
                    
                else:
                    reterr = model.predict(X_train,verbose=0)
                    batchtotAP = 0
                    # for batidx in range(len(cur_label)):    
                    sortederr = [si[0] for si in sorted(enumerate(reterr[0]),reverse=True,key=lambda xy:xy[1])]
                    itemidx = np.where(cur_label[0]==1)
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
                    apseen = apseen + 1
                    curAP = float(totactAP)/apseen
                    
                    if sortederr.index(itemidx[0]) == 0:
                        lastacc = lastacc + 1
                    
                    # batchAP = float(batchtotAP)/len(cur_label)
                    endap = time.time()
                    info = ''
                    info += ' acc = %.2f' % (lastacc/apseen)
                    info += ' batchAP = %.2f' % batchtotAP
                    info += ' curAP = %.2f' % curAP
                    # info += ' read = %.2fs' % ((endread-start))
                    info += ' fit = %.2fs' % ((endfit-start))
                    info += ' calcap = %.2fs' % ((endap-endfit))
                    sys.stdout.write(info)
                    sys.stdout.flush()                
                
                    seen += (MPLC.seen)
                    totloss += MPLC.totals.get('loss')
                    totacc += MPLC.totals.get('acc')
            # vloss = MPLC.history.get('val_loss')
            # vacc = MPLC.history.get('val_acc')
        
        del curRGB, RGBx_trainall, act_labelall, obj_labelall, OFx_trainall, curOF
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
    
    # pdb.set_trace()
    
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
        
        # x_val = np.append(RGBx_val,OFx_val,axis=2)
        x_val = RGBx_val
        cur_label = act_label
        
        # x_val, act_label, obj_label = readjson.getDB(ind=indval[i])
        
        endread = time.time()
        if isGraph:
            score = model.evaluate({'input1':x_val,'actout':act_label,'objout':obj_label},verbose=0)
            # pdb.set_trace()
            totvalloss += score
            totvallen += x_each.shape[0]
        elif ismerge:
            score = model.evaluate({'rgb':RGBx_val,'of':OFx_val,'out':cur_label},verbose=0)
            # pdb.set_trace()
            totvalloss += score
            totvallen += x_each.shape[0]            
        else:
            score = model.evaluate(x_val,cur_label,verbose=0,show_accuracy=True)
            totvalloss += score[0]
            totvalscore += score[1]
            totvallen += x_each.shape[0]            

            
        progbar.update(i)
        if showAP:
            if isGraph:
                reterr = model.predict({'input1':x_val},verbose=0)
                predobj = reterr['objout']
                predact = reterr['actout']
                batchobjAP = 0
                batchactAP = 0
            
                sortedobjerr = [si[0] for si in sorted(enumerate(predobj[0]),reverse=True,key=lambda xy:xy[1])]
                itemobjidx = np.where(obj_label[0]==1)
                # sortederr = [61,32,51,...] ( 0 ~ 154 )
                # itemidx[0] = array([34,51,61]) ( 0 ~ 154 )
                itemobjidx = itemobjidx[0]
                
                sortedacterr = [si[0] for si in sorted(enumerate(predact[0]),reverse=True,key=lambda xy:xy[1])]
                itemactidx = np.where(act_label[0]==1)
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

            elif ismerge:
                reterr = model.predict({'rgb':RGBx_val,'of':OFx_val},verbose=0)
                batchtotAP = 0
                reterr = reterr['out']
                
                sortederr = [si[0] for si in sorted(enumerate(reterr[0]),reverse=True,key=lambda xy:xy[1])]
                itemidx = np.where(cur_label[0]==1)
                # pdb.set_trace()
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
                
                curAP = float(valtotactAP) / (i+1)
                endap = time.time()
                info = ''
                info += ' batchAP = %.2f' % batchtotAP
                info += ' curAP = %.2f' % curAP
                # info += ' read = %.2fs' % ((endread-start))
                # info += ' fit = %.2fs' % ((endfit-start))
                info += ' time = %.2fs' % ((endap-start))
                sys.stdout.write(info)
                sys.stdout.flush()     
                       
            else:
                reterr = model.predict(x_val,verbose=0)
                endpred = time.time()
                batchtotAP = 0
                # for batidx in range(len(cur_label)):    
                sortederr = [si[0] for si in sorted(enumerate(reterr[0]),reverse=True,key=lambda xy:xy[1])]
                itemidx = np.where(cur_label[0]==1)
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
                valapseen = valapseen + 1
                curAP = float(valtotactAP)/valapseen
                if sortederr.index(itemidx[0]) == 0:
                    vallastacc = vallastacc + 1
                # batchAP = float(batchtotAP)/len(cur_label)
                endap = time.time()
                info = ''
                info += ' acc = %.2f' % (vallastacc/valapseen)
                info += ' batchAP = %.2f' % batchtotAP
                info += ' curAP = %.2f' % curAP
                # info += ' read = %.2fs' % ((endread-start))
                info += ' pred = %.2fs' % ((endpred-start))
                info += ' calcap = %.2fs' % ((endap-endpred))
                sys.stdout.write(info)
                sys.stdout.flush()                

    if isGraph:
        tloss.append(totloss/seen)
        vloss.append(totvalloss/len(vals[0]))
        tactap.append(float(totactAP)/apseen)
        tobjap.append(float(totobjAP)/apseen)
        vactap.append(float(valtotactAP)/valapseen)
        vobjap.append(float(valtotobjAP)/valapseen)
        
    elif ismerge:
        tloss.append(totloss/passed)
        vloss.append(totvalloss/len(vals[0]))
        tactap.append(float(totactAP)/passed)
        vactap.append(float(valtotactAP)/len(vals[0]))
        
    else:
        tloss.append(totloss/seen)
        vloss.append(totvalloss/len(vals[0]))
        tacc.append(totacc/seen)
        vacc.append(totvalscore/len(vals[0]))
        tactap.append(float(totactAP)/apseen)
        vactap.append(float(valtotactAP)/len(RGBvals[0]))

    del RGBvals, RGBx_valall, act_labelall, obj_labelall, vals, OFx_valall, OFvals
    gc.collect()
            
    if isGraph:
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
        
    elif ismerge:
        f = open(respath+'/mergelossacc.txt','w')
        for item in tloss:
            f.write("%f,"%item)
        f.write("\n")
        for item in vloss:
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