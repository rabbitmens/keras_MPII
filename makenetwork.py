from keras.models import Sequential, Graph
from keras.layers.core import Dense, Dropout, Activation, MaskedLayer, Flatten, RepeatVector
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.recurrent import LSTM
from keras.layers.embeddings import Embedding
import theano.tensor as T
import pdb

def softmaxdual(x):
    x1 = x
    return T.nnet.softmax(x.reshape((-1, x.shape[-1]))).reshape(x.shape)

class Activations(MaskedLayer):
    '''
        Apply an activation function to an output.
    '''
    def __init__(self, activation, target=0, beta=0.1):
        super(Activations, self).__init__()
        # pdb.set_trace()
        self.activation = softmaxdual
        self.target = target
        self.beta = beta

    def get_output(self, train=False):
        X = self.get_input(train)
        return self.activation(X)

    def get_config(self):
        return {"name": self.__class__.__name__,
                "activation": self.activation.__name__,
                "target": self.target,
                "beta": self.beta}

def makegraph():
    model = Graph()
    model.add_input(name='input1',ndim=3)
    model.add_node(LSTM(4096,2048, return_sequences=True), name='lstm1', input='input1')
    model.add_node(Dropout(0.2),name='drop1',input='lstm1')
    model.add_node(LSTM(2048,512, return_sequences=True), name='lstm2', input='drop1')
    model.add_node(Dropout(0.2),name='drop2',input='lstm2')
    model.add_node(LSTM(512,512, return_sequences=False), name='lstm3', input='drop2')
    model.add_node(Dropout(0.2),name='drop3',input='lstm3')
    model.add_node(Dense(512,67),name='dense1',input='drop3')
    model.add_node(Dense(512,155),name='dense2',input='drop3')
    model.add_node(Activation('softmax'),name='softmax1',input='dense1')
    model.add_node(Activation('softmax'),name='softmax2',input='dense2')
    model.add_output(name='actout',input='softmax1')
    model.add_output(name='objout',input='softmax2')

    return model

def makenetwork():    
    model = Sequential()
    # model.add(Convolution2D(96, 3, 7, 7, border_mode='full')) 
    # model.add(Activation('relu'))
    # # model.add(Convolution2D(96, 96, 7, 7))
    # # model.add(Activation('relu'))
    # model.add(MaxPooling2D(poolsize=(3, 3)))
    # 
    # model.add(Convolution2D(256, 96, 5, 5, border_mode='full')) 
    # model.add(Activation('relu'))
    # # model.add(Convolution2D(256, 256, 5, 5)) 
    # # model.add(Activation('relu'))
    # model.add(MaxPooling2D(poolsize=(3, 3)))
    # 
    # model.add(Convolution2D(512, 256, 3, 3, border_mode='full')) 
    # model.add(Activation('relu'))
    # model.add(MaxPooling2D(poolsize=(3, 3)))
    # 
    # model.add(Convolution2D(512, 512, 3, 3)) 
    # model.add(Activation('relu'))
    # model.add(MaxPooling2D(poolsize=(3, 3)))
    # 
    # model.add(Convolution2D(2048, 512, 3, 3, border_mode='full')) 
    # model.add(Activation('relu'))
    # # model.add(Convolution2D(2048, 2048, 3, 3)) 
    # # model.add(Activation('relu'))
    # model.add(MaxPooling2D(poolsize=(3, 3)))
    # 
    # model.add(Flatten())
    # model.add(Dense(2048, 1024))
    # model.add(Activation('relu'))
    # model.add(Dropout(0.5))
    # 
    # model.add(Dense(1024, 101))
    # model.add(Activation('softmax'))
    
    # model.add(Dense(1024, 512))
    # model.add(Activation('relu'))
    # model.add(Dropout(0.5))
    # 
    # model.add(Dense(1024, 256))
    # model.add(Activation('relu'))
    # model.add(Dropout(0.5))
    
    # # model.add(Embedding(512, 512))
    # model.add(RepeatVector(4096))
    # # the GRU below returns sequences of max_caption_len vectors of size 256 (our word embedding size)
    
    model.add(LSTM(4096, 2048, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(2048, 512, return_sequences=False))
    model.add(Dropout(0.5))
    model.add(Dense(512, 67))
    model.add(Activation('softmax'))
    
    # model.add(LSTM(4096, 4096, return_sequences=True))
    # model.add(Dropout(0.2))
    # model.add(LSTM(4096, 2048, return_sequences=True))
    # model.add(Dropout(0.2))
    # model.add(LSTM(2048, 1024, return_sequences=True))
    # model.add(Dropout(0.2))
    # model.add(LSTM(1024, 512, return_sequences=True))
    # model.add(Dropout(0.2))
    # model.add(LSTM(512, 512, return_sequences=False))
    # # model.add(Dropout(0.2))
    # model.add(Dense(512, 67))
    # model.add(Activation('softmax'))

    return model