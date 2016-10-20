import os
import h5py
import numpy as np
import getpath
from PIL import Image

trainimages = '/home/cvlab/Chenqi/ECCV-Data/images/'
trainlabels = '/home/cvlab/Chenqi/ECCV-Data/labels/'

train_image_name_list = getpath.browser(trainimages)
train_num = len(train_image_name_list)
size = (640,352)

# print train_image_name_list
# raw_input()

for i in range(10):
    training_data_list = []
    training_label_list = []

    for j in range(250*i,250*(i+1)):
        temp_image = Image.open(trainimages + train_image_name_list[j])
        resize_image = temp_image.resize(size, Image.ANTIALIAS)
        resize_image_numpy = np.array(resize_image).reshape(3,352,640)

        temp_label = Image.open(trainlabels + train_image_name_list[j])
        resize_label = temp_label.resize(size, Image.ANTIALIAS)
        resize_label_numpy = np.array(resize_label).reshape(1,352,640)

        # resize_image.show()
        # print resize_image_numpy.shape
        # resize_label.show()
        # print resize_label_numpy.shape
        # raw_input()
        training_data_list.append(resize_image_numpy)
        training_label_list.append(resize_label_numpy)

        print 'Image: ' + str(j)

    h5name = 'h5file/Train' + str(i).zfill(4) + '.h5'

    with h5py.File(h5name,'w') as hf:
        hf.create_dataset('data', dtype = 'f', data = training_data_list)
        hf.create_dataset('label', dtype = 'f', data = training_label_list)
    print h5name + 'Finished'
