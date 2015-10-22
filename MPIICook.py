import numpy as np
import random
import cv2
import os
import itertools
from keras.utils import np_utils
from keras.utils.generic_utils import Progbar
from MPcallback import MPlogcallback
from keras.utils import np_utils
from makenetwork import makenetwork
import readjson
import ujson
import keras.callbacks
import pdb
import os.path
import glob
''' THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32,exception_verbosity=high,optimizer=fast_compile python MPIICook.py '''


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
model = makenetwork()


respath = 'obj'


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

model.compile(loss='categorical_crossentropy', optimizer='rmsprop', class_mode="categorical")

# print('make checkpoint')
# checkpoint = keras.callbacks.ModelCheckpoint(filepath='model.hdf5', verbose=1, save_best_only=True)

MPLC = MPlogcallback()

jsonpath = '/home/rcvbong/jsonsMP'

tloss = []
tacc = []
vloss = []
vacc = []
# pdb.set_trace()
if mval >= 0:
    f = open(respath+'/lossacc.txt','r')
    lines = f.read().splitlines()
    tlossstr = lines[0].split(',')
    taccstr = lines[1].split(',')
    vlossstr = lines[2].split(',')
    vaccstr = lines[3].split(',')
    tlossstr = tlossstr[:len(tlossstr)-1]
    taccstr = taccstr[:len(taccstr)-1]
    vlossstr = vlossstr[:len(vlossstr)-1]
    vaccstr = vaccstr[:len(vaccstr)-1]
    tloss = [float(x) for x in tlossstr]
    tacc = [float(x) for x in taccstr]
    vloss = [float(x) for x in vlossstr]
    vacc = [float(x) for x in vaccstr]
    f.close()
    
    
vlosslist = []
vacclist = []
vseenlist = []
for iteration in range(mval+1,nb_epoch):
    # datapath='/media/disk1/bgsim/Dataset/UCF-101'
    # trainlist,testlist = makeDB(datapath=datapath,divideself=False)
    print(str(iteration)+'th epoch '+'-'*50)
    
    seen = 0
    totloss = 0
    totacc = 0
    progbar = Progbar(target=len(indtrain))
    for i in range(len(indtrain)):
        # pdb.set_trace()
        if not os.path.isfile(os.path.join(jsonpath,str(indtrain[i])+'.json')):
            # print('no file exist')
            continue
        # print('get DB from json files')
        x_train, y_train = readjson.getDB(ind=indtrain[i])
        # y_train = readjson.to_categorical_dual(y_train,10)
        # print(x_train.shape, y_train.shape)
        

        # print('model fit')
        model.fit(x_train,y_train,batch_size=128,nb_epoch=1,show_accuracy=True,shuffle=False,\
                      verbose=0,callbacks=[MPLC])

        # pdb.set_trace()
        progbar.update(i)

        seen += (MPLC.seen)
        totloss += MPLC.totals.get('loss')
        totacc += MPLC.totals.get('acc')
        # vloss = MPLC.history.get('val_loss')
        # vacc = MPLC.history.get('val_acc')


    ### predict validations
    totvalloss = 0
    totvalscore = 0
    totvallen = 0
    print(str(iteration)+'th validation')
    progbar = Progbar(target=len(indval))
    for i in range(len(indval)):
        if not os.path.isfile(os.path.join(jsonpath,str(indval[i])+'.json')):
            # print('no file exist')
            continue
        x_val, y_val = readjson.getDB(ind=indval[i])
        score = model.evaluate(x_val,y_val,verbose=0,show_accuracy=True)
        totvalloss += score[0]
        totvalscore += score[1]
        totvallen += x_val.shape[0]
        progbar.update(i)


    ### save model weights
    model.save_weights(respath+'/model'+str(iteration)+'.hdf5', overwrite=True)
    
    ### draw plot
    from matplotlib import pyplot
    
    tloss.append(totloss/seen)
    tacc.append(totacc/seen)
    vloss.append(totvalloss/len(indval))
    vacc.append(totvalscore/len(indval))
    
    
    vlosslist.append(totvalloss)
    vacclist.append(totvalscore)
    vseenlist.append(totvallen)
    f2 = open('totvalscore.txt','w')
    for item in vlosslist:
        f2.write("%f,"%item)
    f2.write("\n")
    for item in vacclist:
        f2.write("%f,"%item)
    f2.write("\n")
    for item in vseenlist:
        f2.write("%f,"%item)
    f2.write("\n")
    f2.close()
    
    
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
    f.close()
    # pdb.set_trace()
    
    # print(tloss)
    # print(range(len(tloss)))
    # print(min(tloss,vloss).pop(0)-0.5)
    # print([-1.0,len(tloss)+1.0,min(tloss,vloss)[0]-0.5,max(tloss,vloss)[0]+0.5])
    fig = pyplot.figure(1)
    pyplot.subplot(121)
    pyplot.plot(range(len(tloss)),tloss,'k')
    pyplot.plot(range(len(vloss)),vloss,'b')
    pyplot.axis([-1,len(tloss)+1,min([min(tloss),min(vloss)])-0.1,max([max(tloss),max(vloss)])+0.1])
    pyplot.title('loss')
    
    pyplot.subplot(122)
    pyplot.plot(range(len(tacc)),tacc,'k')
    pyplot.plot(range(len(vacc)),vacc,'b')
    pyplot.axis([-1,len(tacc)+1,min([min(tacc),min(vacc)])-0.1,max([max(tacc),max(vacc)])+0.1])
    pyplot.title('acc')
    pyplot.ion()
    pyplot.draw()
    pyplot.savefig((respath+'/LSTM_doing.pdf'))
    pyplot.show()