import numpy as np
import matplotlib.pyplot as plt
import os.path
import scipy
import argparse
import math
import pylab
from sklearn.preprocessing import normalize

caffe_root = '/home/cvlab/caffe-master/'
import sys
sys.path.insert(0, caffe_root + 'python')
import caffe

#Import arguments
parser = argparse.ArgumentParser()
parser.add_argument('-model', type = str, required = True)
parser.add_argument('-weights', type = str, required = True)
parser.add_argument('-iter', type = int, required = True)
args = parser.parser_args()

caffe.set_mode_gpu()

net = caffe.Net(args.model, args.weights, caffe.TEST)

for i in range(0, args.iter):
    net.forward()

    image = net.blobs['data'].data
    label = net.blobs['label'].data
    predict = net.blobs['prob'].data

    image = np.squeeze(image[0,:,:,:])
    output = np.squeeze(predict[0,:,:,:])
    ind = np.argmax(output,axis = 0)

#ind is the predicted label,transform ind to colorful image according to the label
#Example:

    # Sky = [128,128,128]
	# Building = [128,0,0]
	# Pole = [192,192,128]
	# Road_marking = [255,69,0]
	# Road = [128,64,128]
	# Pavement = [60,40,222]
	# Tree = [128,128,0]
	# SignSymbol = [192,128,128]
	# Fence = [64,64,128]
	# Car = [64,0,128]
	# Pedestrian = [64,64,0]
	# Bicyclist = [0,128,192]
	# Unlabelled = [0,0,0]
    #
	# label_colours = np.array([Sky, Building, Pole, Road, Pavement, Tree, SignSymbol, Fence, Car, Pedestrian, Bicyclist, Unlabelled])
	# for l in range(0,11):
	# 	r[ind==l] = label_colours[l,0]
	# 	g[ind==l] = label_colours[l,1]
	# 	b[ind==l] = label_colours[l,2]
	# 	r_gt[label==l] = label_colours[l,0]
	# 	g_gt[label==l] = label_colours[l,1]
	# 	b_gt[label==l] = label_colours[l,2]

### Visualizing the Network Graph ###

# from google.protobuf import text_format
# from caffe.draw import get_pydot_graph
# from caffe.proto import caffe_pb2
# from IPython.display import display, Image
#
# _net = caffe_pb2.NetParameter()
# f = open("train.prototxt")
# text_format.Merge(f.read(), _net)
# display(Image(get_pydot_graph(_net, "TB").create_png()))
