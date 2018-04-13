import keras
import h5py
import os
import argparse

parser = argparse.ArgumentParser(description='Re-train AdaNet model')

parser.add_argument('feature_loc', metavar='F', type=str, 
                   help='location of the FBank feature h5 file')
parser.add_argument('annot_loc', metavar='A', type=str, 
                   help='location of the annotation h5 file')
                   
parser.add_argument('set', metavar='S', type=str, 
                   help='location of the annotation h5 file')
                                      
                                    
parser.add_argument('--model_loc', default='models/model_final',
                   help='location of the model. If left empty, new model is created')                                      
parser.add_argument('--model_target', default='models/model_final',
                   help='location of the model. If left empty, new model is created')
parser.add_argument('-e', default=10,
                   help='Number of training epochs')
parser.add_argument('-b', default=100,
                   help='Batch size')

parser.add_argument('-lr', default=0.0001,
                   help='Learning rate')


args = parser.parse_args()


for letter in ('L','R','N'):
    print(letter)
    
    model=keras.models.load_model(args.model_loc)
    model.optimizer = keras.optimizers.SGD(lr=args.lr)

    X = h5py.File(args.feature_loc,'r')
    T = h5py.File(args.annot_loc,'r')
    
    j = 0
    while os.path.exists(args.model_target+"%s" % j):
        j += 1

    available_name = (args.model_target+str(j))

    H = model.fit(X[args.set],T[args.set],epochs=args.e, batch_size=args.b, shuffle='batch')
    model.save(available_name)
       
