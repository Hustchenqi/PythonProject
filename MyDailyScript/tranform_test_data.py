import os
import h5py
from PIL import Image
import numpy as np
import getpath

trainimage = '/home/cvlab/Chenqi/Python/python/train/'
trainlabel = '/home/cvlab/Chenqi/Python/python/trainannot'
#precalc_path = '/home/cvlab/Chenqi/Python/python/trainMaskmultiple'
trainresizedlabel = ['/home/cvlab/Chenqi/Python/python/trainresizedlabel/label1/',
                     '/home/cvlab/Chenqi/Python/python/trainresizedlabel/label2/',
                     '/home/cvlab/Chenqi/Python/python/trainresizedlabel/label3/',
                     '/home/cvlab/Chenqi/Python/python/trainresizedlabel/label4/',
                     '/home/cvlab/Chenqi/Python/python/trainresizedlabel/label5/']

testimage = '/home/cvlab/Chenqi/Python/python/Test/'
testlabel = '/home/cvlab/Chenqi/Python/python/Testannot'
#testMasklabel = '/home/cvlab/Chenqi/Python/python/TestMaskmultiple'
testresizedlabel = ['/home/cvlab/Chenqi/Python/python/Testresizedlabel/label1/',
                    '/home/cvlab/Chenqi/Python/python/Testresizedlabel/label2/',
                    '/home/cvlab/Chenqi/Python/python/Testresizedlabel/label3/',
                    '/home/cvlab/Chenqi/Python/python/Testresizedlabel/label4/',
                    '/home/cvlab/Chenqi/Python/python/Testresizedlabel/label5/']

precalc_path = '/home/cvlab/Chenqi/Python/python/Transfromedprecalc/'

train_label_list = getpath.browser(trainlabel)
#train_mask_label_list = getpath.browser(precalc_path)
train_name_list = train_label_list

test_label_list = getpath.browser(testlabel)
#test_mask_label_list = getpath.browser(testMasklabel)
test_name_list = test_label_list

train_num = len(train_label_list)
test_num = len(test_label_list)

training_data_list = [[] for i in range(7)]
testing_data_list = [[] for i in range(7)]

################################################################################
## STEP ONE
#### save training masked label
# for i in range(train_num):
#     full_unchanged_label_name = trainlabel + '/' + train_label_list[i]
#     new_saved_label_name = precalc_path + '/' + train_label_list[i]
#     temp_result = getpath.image_mask(full_unchanged_label_name)
#     img = Image.fromarray(temp_result, 'L')
#     img.save(new_saved_label_name)
#### save testing masked label
# for i in range(test_num):
#     full_unchanged_label_name = testlabel + '/' + test_name_list[i]
#     new_saved_label_name = testMasklabel + '/' + test_name_list[i]
#     temp_result = getpath.image_mask(full_unchanged_label_name)
#     img = Image.fromarray(temp_result, 'L')
#     img.save(new_saved_label_name)

################################################################################
# STEP TWO
#### save training resized label
if os.path.isfile('trainresizedlabel/label1/lagr_DS1A001.png') == False:
    print 'Processing training data'
    for j in range(train_num):
        inputpath = precalc_path + train_name_list[j]
        filename = train_name_list[j]
        outputpath = trainresizedlabel
        getpath.image_resize(inputpath, filename, outputpath)
        print 'Training Image: ' + filename
    #### save testing resized label
    print 'Processing testing data'
    for j in range(test_num):
        inputpath = precalc_path + test_name_list[j]
        filename = test_name_list[j]
        outputpath = testresizedlabel
        getpath.image_resize(inputpath, filename, outputpath)
        print 'Testing Image: ' + filename

    raw_input('Data Procesing Completed! Pressing Any Key to The Next Step!')
################################################################################
# STEP THREE
#### write training data into HD5 file
for i in range(train_num):

    temp_image = np.array(Image.open(trainimage + train_name_list[i])).reshape(3,480,640)
    temp_label = np.array(Image.open(trainlabel + '/' + train_name_list[i])).reshape(1,480,640)
    temp_label1 = np.array(Image.open(trainresizedlabel[0] + train_name_list[i])).reshape(1,480,640)
    temp_label2 = np.array(Image.open(trainresizedlabel[1] + train_name_list[i])).reshape(1,240,320)
    temp_label3 = np.array(Image.open(trainresizedlabel[2] + train_name_list[i])).reshape(1,120,160)
    temp_label4 = np.array(Image.open(trainresizedlabel[3] + train_name_list[i])).reshape(1,60,80)
    temp_label5 = np.array(Image.open(trainresizedlabel[4] + train_name_list[i])).reshape(1,30,40)

    training_data_list[0].append(temp_image)
    training_data_list[1].append(temp_label)
    training_data_list[2].append(temp_label1)
    training_data_list[3].append(temp_label2)
    training_data_list[4].append(temp_label3)
    training_data_list[5].append(temp_label4)
    training_data_list[6].append(temp_label5)

    print 'Saving Training Data: '+ str(i)
#### write testing data into HD5 file
for k in range(test_num):

    temp_image = np.array(Image.open(testimage + test_name_list[k])).reshape(3,480,640)
    temp_label = np.array(Image.open(testlabel + '/' + test_name_list[k])).reshape(1,480,640)
    temp_label1 = np.array(Image.open(testresizedlabel[0] + test_name_list[k])).reshape(1,480,640)
    temp_label2 = np.array(Image.open(testresizedlabel[1] + test_name_list[k])).reshape(1,240,320)
    temp_label3 = np.array(Image.open(testresizedlabel[2] + test_name_list[k])).reshape(1,120,160)
    temp_label4 = np.array(Image.open(testresizedlabel[3] + test_name_list[k])).reshape(1,60,80)
    temp_label5 = np.array(Image.open(testresizedlabel[4] + test_name_list[k])).reshape(1,30,40)

    testing_data_list[0].append(temp_image)
    testing_data_list[1].append(temp_label)
    testing_data_list[2].append(temp_label1)
    testing_data_list[3].append(temp_label2)
    testing_data_list[4].append(temp_label3)
    testing_data_list[5].append(temp_label4)
    testing_data_list[6].append(temp_label5)

    print 'Saving Testing Data: ' + str(k)

raw_input('Start to Making H5 File, Press Any Key to Continue!')
################################################################################
# STEP FOUR
#### generate training.h5 File
with h5py.File('training.h5', 'w') as hf:
    hf.create_dataset('data', dtype = 'f', data = training_data_list[0])
    hf.create_dataset('label', dtype = 'f', data = training_data_list[1])
    hf.create_dataset('label1', dtype = 'f', data = training_data_list[2])
    hf.create_dataset('label2', dtype = 'f', data = training_data_list[3])
    hf.create_dataset('label3', dtype = 'f', data = training_data_list[4])
    hf.create_dataset('label4', dtype = 'f', data = training_data_list[5])
    hf.create_dataset('label5', dtype = 'f', data = training_data_list[6])
#### generate testing.h5 file
with h5py.File('testing.h5', 'w') as hf:
    hf.create_dataset('data', dtype = 'f', data = testing_data_list[0])
    hf.create_dataset('label', dtype = 'f', data = testing_data_list[1])
    hf.create_dataset('label1', dtype = 'f', data = testing_data_list[2])
    hf.create_dataset('label2', dtype = 'f', data = testing_data_list[3])
    hf.create_dataset('label3', dtype = 'f', data = testing_data_list[4])
    hf.create_dataset('label4', dtype = 'f', data = testing_data_list[5])
    hf.create_dataset('label5', dtype = 'f', data = testing_data_list[6])
print 'Success'
