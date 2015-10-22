import ujson
import os
import numpy as np
from keras.utils import np_utils
import pdb
# f = open("Subject1,arranging_objects.json",'r')
# json = ujson.loads(f.read())
# f.close()
# shell = json.values()
# data = shell[0]
# fda = data[0];
# print(fda.keys()) #  sub, act, rep, name, activ, labact, labojb
# #order : labact, labobj, sub, rep, activ, act, name
# 
# print(fda['name'])
def to_categorical_dual(y, nb_classes):
    '''Convert class vector (integers from 0 to nb_classes)
    to binary class matrix, for use with categorical_crossentropy
    '''
    y = np.asarray(y, dtype='int32')

    Y = np.zeros((len(y), nb_classes))
    for i in range(len(y)):
        Y[i, y[i,0]] = 1
        # Y[i, (y[i,1]+10)] = 1
    return Y

def getDB(jsonpath='/home/rcvbong/jsonsMP',ind=1):
    
    listactiv = []
    listfnum = []    

    # pdb.set_trace()

    f = open(os.path.join(jsonpath,str(ind)+'.json'),'r')
    json = ujson.loads(f.read())
    f.close()
    shell = json.values()
    data = shell[0]
    labels = data[0]
    # labact = labels['labact']
    labobj = labels['labobj']
    
    first = data[1]
    activ = first['activ']
    X_train = np.zeros((len(data)-1, len(activ),1), dtype="float")
    
    for i in range(1,len(data)):
        datum = data[i]
        X_train[i-1,:] = np.array(datum['activ'])
        # listactiv.append(datum['activ'])
        # listfnum.append(datum['fnum'])
        
    # pdb.set_trace()
        
    # print(len(listactiv[0]),len(listlabact),len(listlabobj))
    
    # for ind in range(len(listactiv)):
    #     X_train[ind,:] = np.array(listactiv[ind])
    # X_train = np.squeeze(X_train)
    # print(X_train.shape)
    X_train = X_train.transpose(0,2,1)
    # print(X_train.shape)
    # Y_train = np.zeros((len(data)-1, 222), dtype="uint8")
    Y_train = np.zeros((len(data)-1, 155), dtype="uint8")
    
    # Y_train[:,labact-1] = 1
    Y_train[:,:] = labobj
    # Y_train[:,67:222] = labobj
    
    # for ind in range(len(listlabact)):
    #     Y_train[ind,0] = (listlabact[ind]-1)
    #     Y_train[ind,1] = (listlabobj[ind]-1)
    
    
    # X_test = np.zeros((len(testactiv), len(testactiv[0]),1), dtype="float")
    # for ind in range(len(testactiv)):
    #     X_test[ind,:] = np.array(testactiv[ind])
    # # X_train = np.squeeze(X_train)
    # print(X_test.shape)
    # X_test = X_test.transpose(0,2,1)
    # print(X_test.shape)
    # Y_test = np.zeros((len(testactiv), 2), dtype="uint8")
    # for ind in range(len(testlabact)):
    #     Y_test[ind,0] = (testlabact[ind]-1)
    #     Y_test[ind,1] = (testlabobj[ind]-1)
        
    # print(Y_train[0])
    return X_train, Y_train

