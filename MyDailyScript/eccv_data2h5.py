import os
import h5py
import numpy as np
import getpath
from PIL import Image

trainimages = '/home/cvlab/Chenqi/ECCV-Data/images/'
trainlabels = '/home/cvlab/Chenqi/ECCV-Data/labels/'

train_image_name_list = getpath.browser(trainimages)
train_num = len(train_image_name_list)

for i in range(5):
    training_data_list = []
    training_label_list = []

    for j in range(50*i,50*(i+1)):
        temp_image = np.array(Image.open(trainimages + train_image_name_list[j])).reshape(3,1052,1914)
        temp_label = np.array(Image.open(trainlabels + train_image_name_list[j])).reshape(1,1052,1914)

        training_data_list.append(temp_image)
        training_label_list.append(temp_label)

        print 'Image: ' + str(j)

    h5name = 'h5file/Train' + str(i).zfill(4) + '.h5'
    
    with h5py.File(h5name,'w') as hf:
        hf.create_dataset('data', dtype = 'f', data = training_data_list)
        hf.create_dataset('label', dtype = 'f', data = training_label_list)
    print h5name + 'Finished'
