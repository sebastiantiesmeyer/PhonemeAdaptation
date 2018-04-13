##AdaNet network specification.
from __future__ import print_function

import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, BatchNormalization
from keras.optimizers import Adam,SGD
from keras.callbacks import EarlyStopping
from keras.models import load_model

import os
import h5py

#Define a few variables that will be utilised throughout the script:




input_count = 64*11
output_count = 39

class AnaNet():
    '''
    Network class for a DNN
    
    '''
    def __init__(self,input_count = (11,64,),batch_size = 100, hidden_count = 6):
        self.input_count = input_count
        self.batch_size = batch_size
        self.model = self.build(hidden_count=hidden_count)
        
        
    def build(self, hidden_count):
        
        model = Sequential()
        model.add(Dense(units = input_count, input_shape = [11,64,], activation = 'linear'))
        model.add(Flatten())
        for i in range(hidden_count-1):
            model.add(Dense(1024, activation = 'sigmoid'))
            model.add(BatchNormalization())
            model.add(Dropout(0.1))
            
        model.add(Dense(output_count, activation = 'sigmoid', name='preds'))
        
        optimizer = Adam() # SGD(lr = 0.001, decay = 0, momentum = 0)#momentum = 0.9, decay=1e-6, nesterov = True)
        
        model.compile(optimizer = optimizer, loss = 'categorical_crossentropy',metrics=['accuracy'])
        
        return(model)
        
    def fit(self, X,T):
        
        self.model.fit(X,T)
        
    def fit_generator(self, generator, val_generator, steps = [20000,1000],nb_epoch = 50 ):
        
        early = EarlyStopping(monitor='val_loss', min_delta=0, patience=25, verbose=1, mode='auto')

        self.model.fit_generator(generator.__getitem__(), steps_per_epoch=steps[0]//self.batch_size, class_weight = None, verbose = 1, use_multiprocessing = False, epochs = nb_epoch,max_queue_size=1, workers=1,  callbacks=[early],validation_steps=steps[1]//self.batch_size, validation_data=val_generator.__getitem__())

        
    def predict(self,X):
        
        return(self.model.predict(X))
       
class kaldi_generator():
    def __init__(self, batch_size = 100,
                data_loc = '/media/sebastian/7B4861FD6D0F6AA2/finalt.h5',
                feature_loc = '/media/sebastian/7B4861FD6D0F6AA2/finalx.h5',
                file_keys = None, val_perc = 0, validation = False):
                    
        self.batch_size = batch_size            
        self.n = 0
        self.X = h5py.File(feature_loc,'r')
        self.T = h5py.File(data_loc,'r')
        self.X_stats = {}
        self.counter = 0
        
        if file_keys:
            self.X_keys = file_keys
        else:
            self.X_keys = [k for k in self.X.keys()]
            val_count = int(len(self.X_keys)*val_perc)
        
            if validation:
                self.X_keys = self.X_keys[:val_count]
            else:
                self.X_keys = self.X_keys[val_count:]
        
        for k in self.X.keys():
            self.X_stats[k]=[np.random.randint(self.X[k].shape[0]-20), self.X[k].shape[0]-20,0]
        
    def __getitem__(self):
        
        
        while 1:
            
            X = np.zeros((self.batch_size,11,64))
            T = np.zeros((self.batch_size,39))
            for i in range(0,self.batch_size,100):

                if self.n==len(self.X_keys): self.n=0  
                    
                key = self.X_keys[self.n]
                if self.X_stats[key][0]+100 >= self.X_stats[key][1]:
                    self.X_stats[key][0] = 0
                
                X[i:i+100,:,:] = self.X[key][self.X_stats[key][0]:self.X_stats[key][0]+100,:,:]        
                T[i:i+100,:] = self.T[key][self.X_stats[key][0]:self.X_stats[key][0]+100,:]
                
                self.X_stats[key][0]+=100
                self.n += 1
                
            yield (X,T)



#gen = kaldi_generator(file_keys = train_files, val_perc = 0.1, validation = False)#train_files)
#val_gen = kaldi_generator(file_keys = val_files, val_perc = 0.1, validation = True)# test_files)

def __main__(args):
    
    split_loc = 'resources/split.npy'

    splits = np.load(split_loc,'r')
    val_files = [i[0] for i in splits if i[1]=='val']
    train_files = [i[0] for i in splits if i[1]=='train']
    test_files = [i[0] for i in splits if i[1]=='test']

    X = h5py.File(args.feature_loc,'r')
    T = h5py.File(args.annot_loc,'r')
    
    net = AnaNet(batch_size = args.batch_size)
    if args.model_loc:
        net.model = load_model(args.model_loc)

    for i in range(args.epochs):
        for tset in train_files:
            print(i,tset)
            net.model.fit(X[tset],T[tset],batch_size = args.batch_size, shuffle='batch')
        dpoints = 0    
        loss = 0
        
        print('validating...')
        for vset in val_files:
            length = T[vset].shape[0]
            dpoints+= length
            loss += net.model.evaluate(X[vset],T[vset],batch_size = args.batch_size,verbose = 0)[0]*length
        print(loss/dpoints)   
        
        j = 0
        while os.path.exists(args.model_target+"%s" % j):
            j += 1

        available_name = (args.model_target+str(j))
        
        net.model.save(args.model_target+str(j))


























