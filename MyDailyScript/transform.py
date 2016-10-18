import os
import h5py
from PIL import Image
import numpy as np
import getpath

path_groundtruth = '/home/cvlab/SegNet/CamVidB/Trainannot'
path_train = '/home/cvlab/SegNet/CamVidB/Train'
path_new = '/home/cvlab/Chenqi/Python/python/Transformedimage'
path_precalc = '/home/cvlab/Chenqi/Python/python/precalc'
path_transform_precalc = '/home/cvlab/Chenqi/Python/python/Transfromedprecalc/'
# finetune_path_groundtruth = '/home/cvlab/SegNet/CamVidB/Testannot'
# finetune_path_new = '/home/cvlab/Chenqi/Python/python/Testannot'

path_test_data = '/home/cvlab/SegNet/CamVidB/Val'
path_test_label = '/home/cvlab/SegNet/CamVidB/Valannot'
test_data_name = getpath.browser(path_test_data)
test_num = len(test_data_name)

#finetune_groundtruth_name  = getpath.browser(finetune_path_groundtruth)
groundtruth_name = getpath.browser(path_groundtruth)
train_name = getpath.browser(path_train)
precalc_list = getpath.browser(path_precalc)

img_num = len(groundtruth_name)
# fine_num = len(finetune_groundtruth_name)

ground_img = []
train_img = []
test_data_list = []
test_label_list = []
################################################################################
#### Transform precalc label
for element in precalc_list:
    precalc_read = np.array(Image.open(path_precalc + '/' + element))
    precalc_new = getpath.elementmodify(precalc_read)
    img = Image.fromarray(precalc_new, 'L')
    img.save(path_transform_precalc + element)
    print 'Transforming: ' + element

################################################################################
#### Transform finetune groundtruth label
# for i in range(fine_num):
#     fine_ground = np.array(Image.open(finetune_path_groundtruth + '/' +finetune_groundtruth_name[i]))
#     ine_new_ground = getpath.elementmodify(fine_ground)
#     img = Image.fromarray(fine_new_ground, 'L')
#     img.save(finetune_path_new+'/'+finetune_groundtruth_name[i])
#     print i

################################################################################
#### transform training groundtruth label
# for i in range(img_num):
#     temp_ground = np.array(Image.open(path_groundtruth + '/' +groundtruth_name[i]))
#     new_ground = getpath.elementmodify(temp_ground)
#     img = Image.fromarray(new_ground, 'L')
#     img.save(path_new+'/'+groundtruth_name[i])
#     print i

################################################################################
#### making training data
#for i in range(img_num):
    # new_ground = np.array(Image.open(path_new + '/' + groundtruth_name[i])).reshape(1,480,640)
    # temp_img = np.array(Image.open(path_train + '/' + train_name[i])).reshape(3,480,640)
    # ground_img.append(new_ground)
    # train_img.append(temp_img)
    #print i

#######example
    # temp_img = np.array(Image.open(path_train + '/' + train_name[i]))
    # sample1 = temp_img.reshape(3,480,640)
    # sample2 = temp_img.transpose()
    # sample3 = temp_img.transpose((2,0,1))
    #
    # print temp_img.shape
    # print sample1.shape
    # print sample2.shape
    # print sample3.shape
    #
    # if sample3.all() == sample1.all() :
    #     print 'true'
    # raw_input()

#
# with h5py.File('train.h5', 'w') as hf:
#     hf.create_dataset('data',dtype='f',data=train_img)
#     hf.create_dataset('label',dtype='f',data=ground_img)

################################################################################
#### making testing data
# for i in range(test_num):
#     temp_data = np.array(Image.open(path_test_data + '/' + test_data_name[i])).reshape(3,480,640)
#     temp_label = np.array(Image.open(path_test_label + '/' + test_data_name[i])).reshape(1,480,640)
#     test_data_list.append(temp_data)
#     test_label_list.append(temp_label)
#     print 'Iteration ' + str(i)
#
# with h5py.File('test.h5', 'w') as hf:
#     hf.create_dataset('data', dtype = 'f', data = test_data_list)
#     hf.create_dataset('label', dtype = 'f', data = test_label_list)
