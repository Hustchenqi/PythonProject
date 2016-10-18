import os
import scipy
import numpy as np
from PIL import Image


def browser(path):
    if not os.path.isdir(path):
        return

    for root, dirs, lists in os.walk(path):
        return lists

def elementmodify(nparray):
    for i in range(480):
        for j in range(640):
            if nparray[i][j] == 0:
                nparray[i][j] = 2
            elif nparray[i][j] == 2:
                nparray[i][j] =0
    return nparray

def image_mask(imagepath):
    nparray = np.array(Image.open(imagepath))
    mask = np.zeros([480,640], dtype = np.uint8)

    #define measurable block
    mask[240:480,0:640] = 1
    result = np.multiply(nparray, mask)
    return result

def image_resize(inputpath,filename,outputpath):
    img = Image.open(inputpath, 'r')
    label1 = img.resize((640,480), Image.ANTIALIAS)
    label2 = img.resize((320,240), Image.ANTIALIAS)
    label3 = img.resize((160,120), Image.ANTIALIAS)
    label4 = img.resize((80,60), Image.ANTIALIAS)
    label5 = img.resize((40,30), Image.ANTIALIAS)
    label1.save(outputpath[0]+filename)
    label2.save(outputpath[1]+filename)
    label3.save(outputpath[2]+filename)
    label4.save(outputpath[3]+filename)
    label5.save(outputpath[4]+filename)

def matrix_resize(inputmatrix):
    temp_input_matrix = np.array(inputmatrix, dtype = float)
    print temp_input_matrix
    disparity1 = scipy.misc.imresize(temp_input_matrix,(640,480))
    disparity2 = scipy.misc.imresize(temp_input_matrix,(320,240))
    disparity3 = scipy.misc.imresize(temp_input_matrix,(160,120))
    disparity4 = scipy.misc.imresize(temp_input_matrix,(80,60))
    disparity5 = scipy.misc.imresize(temp_input_matrix,(40,30))

    return [disparity1,disparity2,disparity3,disparity4,disparity5]

def split_label(test_label_path):
    test_label = np.array(Image.open(test_label_path))
    label1_1 = np.zeros(test_label.shape)
    label1_2 = np.zeros(test_label.shape)
    for m in range(test_label.shape[0]):
        for n in range(test_label.shape[1]):
            if test_label[m][n] == 2:
                label1_2[m][n] = 1
            elif test_label[m][n] == 1:
                label1_1[m][n] = 1
    return label1_1, label1_2

def modify_label(test_label_path):
    test_label = np.array(Image.open(test_label_path))
    label = np.zeros(test_label.shape)
    for m in range(test_label.shape[0]):
        for n in range(test_label.shape[1]):
            if test_label[m][n] == 2:
                label[m][n] = 1
    return label
#browser('/home/chenqi/Trainannot')
