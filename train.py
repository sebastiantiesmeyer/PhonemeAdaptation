import argparse
import AdaNet

parser = argparse.ArgumentParser(description='Create & train AdaNet model')

parser.add_argument('feature_loc', metavar='F', type=str, nargs='1',
                   help='location of the FBank feature h5 file')
parser.add_argument('annot_loc', metavar='A', type=str, nargs='1',
                   help='location of the annotation h5 file')
                                      
parser.add_argument('model_loc', default=None,
                   help='location of the model. If left empty, new model is created')                                      
parser.add_argument('model_target', default=None,
                   help='location of the model. If left empty, new model is created')
parser.add_argument('-e', default=10,
                   help='Number of training epochs')

parser.add_argument('-b', default=100,
                   help='Batch size')


args = parser.parse_args()

AdaNet.__main__(args)

