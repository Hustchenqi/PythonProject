import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mping
import os.path
import json
import scipy
import argparse
import math
import pylab
from sklearn.preprocessing import normalize
from PIL import Image

import sys

def label2rgb(imagepath, savepath):
    label = np.array(Image.open(imagepath))
    print label[300][300]
    r = label.copy()
    g = label.copy()
    b = label.copy()

    ground = [0,0,128]
    obstacle = [0,128,0]
    unlabelled = [128,0,0]
    label_colors = np.array([unlabelled,obstacle,ground])
    l = 0
    r[label == l] = label_colors[l,0]
    g[label == l] = label_colors[l,1]
    b[label == l] = label_colors[l,2]
    l = 255
    r[label == l] = label_colors[2,0]
    g[label == l] = label_colors[2,1]
    b[label == l] = label_colors[2,2]
    rgb = np.zeros((label.shape[0],label.shape[1],3))
    rgb[:,:,0] = r
    rgb[:,:,1] = g
    rgb[:,:,2] = b
    scipy.misc.toimage(rgb, cmin = 0.0, cmax = 255).save(savepath)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type = str, required = True)
    parser.add_argument('--save', type = str, required = True)
    args = parser.parse_args()

    label2rgb(args.image, args.save)
    print 'success'
