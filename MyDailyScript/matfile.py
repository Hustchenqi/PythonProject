import os
import h5py
from PIL import Image
import numpy as np
import scipy.io as sio
import getpath

training_data_list = [[] for i in range(7)]
testing_data_list = [[] for i in range(7)]

mat_contents = sio.loadmat('resized_disparity.mat')
#print mat_contents
label1 = mat_contents['label1_mat']
label2 = mat_contents['label2_mat']
label3 = mat_contents['label3_mat']
label4 = mat_contents['label4_mat']
label5 = mat_contents['label5_mat']
# print label5[:,:,1]
image = '/home/cvlab/Chenqi/Python/python/Image'
label = '/home/cvlab/Chenqi/Python/python/Label'

image_name_list = getpath.browser(image)
label_name_list = getpath.browser(label)

image_name_list.sort()
label_name_list.sort()
# print image_name_list == label_name_list
# raw_input()
mid = 500    #define the middle number to devide dataset into two parts

for i in range(0,600):
    if i in range(mid,mid+100):
        test_image = np.array(Image.open(image + '/' + image_name_list[i])).reshape(3,480,640)
        test_label= getpath.modify_label(label + '/' + label_name_list[i])
        test_label1_modified = test_label.reshape(1,480,640)
        test_label1 = np.array(label1[:,:,i]).reshape(1,480,640)
        test_label2 = np.array(label2[:,:,i]).reshape(1,240,320)
        test_label3 = np.array(label3[:,:,i]).reshape(1,120,160)
        test_label4 = np.array(label4[:,:,i]).reshape(1,60,80)
        test_label5 = np.array(label5[:,:,i]).reshape(1,30,40)

        testing_data_list[0].append(test_image)
        testing_data_list[1].append(test_label1_modified)
        testing_data_list[2].append(test_label1)
        testing_data_list[3].append(test_label2)
        testing_data_list[4].append(test_label3)
        testing_data_list[5].append(test_label4)
        testing_data_list[6].append(test_label5)

        print 'Saving Testing Data: ' + str(i)
    else:
        train_image = np.array(Image.open(image + '/' + image_name_list[i])).reshape(3,480,640)
        train_label= getpath.modify_label(label + '/' + label_name_list[i])
        train_label1_modified = train_label.reshape(1,480,640)
        train_label1 = np.array(label1[:,:,i]).reshape(1,480,640)
        train_label2 = np.array(label2[:,:,i]).reshape(1,240,320)
        train_label3 = np.array(label3[:,:,i]).reshape(1,120,160)
        train_label4 = np.array(label4[:,:,i]).reshape(1,60,80)
        train_label5 = np.array(label5[:,:,i]).reshape(1,30,40)

        training_data_list[0].append(train_image)
        training_data_list[1].append(train_label1_modified)
        training_data_list[2].append(train_label1)
        training_data_list[3].append(train_label2)
        training_data_list[4].append(train_label3)
        training_data_list[5].append(train_label4)
        training_data_list[6].append(train_label5)
        print 'Saving Training Data: '+ str(i)

raw_input('Start to making H5 File, Press Any Key to Continue!')

with h5py.File('training.h5', 'w') as hf:
    hf.create_dataset('data', dtype = 'f', data = training_data_list[0])
    hf.create_dataset('label', dtype = 'f', data = training_data_list[1])
    # hf.create_dataset('label1_2', dtype = 'f', data = training_data_list[2])
    hf.create_dataset('label1', dtype = 'f', data = training_data_list[2])
    hf.create_dataset('label2', dtype = 'f', data = training_data_list[3])
    hf.create_dataset('label3', dtype = 'f', data = training_data_list[4])
    hf.create_dataset('label4', dtype = 'f', data = training_data_list[5])
    hf.create_dataset('label5', dtype = 'f', data = training_data_list[6])
#### generate testing.h5 file
with h5py.File('testing.h5', 'w') as hf:
    hf.create_dataset('data', dtype = 'f', data = testing_data_list[0])
    hf.create_dataset('label', dtype = 'f', data = testing_data_list[1])
    # hf.create_dataset('label1_2', dtype = 'f', data = testing_data_list[2])
    hf.create_dataset('label1', dtype = 'f', data = testing_data_list[2])
    hf.create_dataset('label2', dtype = 'f', data = testing_data_list[3])
    hf.create_dataset('label3', dtype = 'f', data = testing_data_list[4])
    hf.create_dataset('label4', dtype = 'f', data = testing_data_list[5])
    hf.create_dataset('label5', dtype = 'f', data = testing_data_list[6])
print 'Success'
