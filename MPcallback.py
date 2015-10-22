import keras
import keras.callbacks

class MPlogcallback(keras.callbacks.Callback):
    
    def __init__(self, monitor='val_loss', patience=0, verbose=0):
        super(keras.callbacks.Callback, self).__init__()
        
        
    def on_epoch_begin(self, epoch, logs={}):
        self.seen = 0
        self.history = {}
        self.totals = {}

    def on_batch_end(self, batch, logs={}):
        batch_size = logs.get('size', 0)
        self.seen += batch_size

        for k, v in logs.items():
            if k in self.totals:
                self.totals[k] += v * batch_size
            else:
                self.totals[k] = v * batch_size

    def on_epoch_end(self, epoch, logs={}):
        
        for k, v in self.totals.items():
            if k not in self.history:
                self.history[k] = []
            self.history[k].append(v / self.seen)

        for k, v in logs.items():
            if k not in self.history:
                self.history[k] = []
            self.history[k].append(v)
                
        # print(self.history)
        # 
        # from matplotlib import pyplot
        # 
        # tloss = self.history.get('loss')
        # tacc = self.history.get('acc')
        # vloss = self.history.get('val_loss')
        # vacc = self.history.get('val_acc')
        # 
        # # print(tloss)
        # # print(range(len(tloss)))
        # # print(min(tloss,vloss).pop(0)-0.5)
        # # print([-1.0,len(tloss)+1.0,min(tloss,vloss)[0]-0.5,max(tloss,vloss)[0]+0.5])
        # fig = pyplot.figure(1)
        # pyplot.subplot(121)
        # pyplot.plot(range(len(tloss)),tloss,'k')
        # pyplot.plot(range(len(vloss)),vloss,'b')
        # pyplot.axis([-1,len(tloss)+1,min(tloss,vloss)[0]-0.1,max(tloss,vloss)[0]+0.1])
        # pyplot.title('loss')
        # 
        # pyplot.subplot(122)
        # pyplot.plot(range(len(tacc)),tacc,'k')
        # pyplot.plot(range(len(vacc)),vacc,'b')
        # pyplot.axis([-1,len(tacc)+1,min(tacc,vacc)[0]-0.1,max(tacc,vacc)[0]+0.1])
        # pyplot.title('acc')
        # pyplot.ion()
        # pyplot.draw()
        # pyplot.savefig(('epoch%03d.pdf'%epoch))
        # # trainloss = logs.get('train_loss', 0)
