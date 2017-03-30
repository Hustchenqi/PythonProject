import os
import numpy as np
import cv2
from collections import namedtuple

Label = namedtuple( 'Label' , [

    'name'        , # The identifier of this label, e.g. 'car', 'person', ... .
                    # We use them to uniquely name a class

    'id'          , # An integer ID that is associated with this label.
                    # The IDs are used to represent the label in ground truth images
                    # An ID of -1 means that this label does not have an ID and thus
                    # is ignored when creating ground truth images (e.g. license plate).
                    # Do not modify these IDs, since exactly these IDs are expected by the
                    # evaluation server.

    'trainId'     , # Feel free to modify these IDs as suitable for your method. Then create
                    # ground truth images with train IDs, using the tools provided in the
                    # 'preparation' folder. However, make sure to validate or submit results
                    # to our evaluation server using the regular IDs above!
                    # For trainIds, multiple labels might have the same ID. Then, these labels
                    # are mapped to the same class in the ground truth images. For the inverse
                    # mapping, we use the label that is defined first in the list below.
                    # For example, mapping all void-type classes to the same ID in training,
                    # might make sense for some approaches.
                    # Max value is 255!

    'category'    , # The name of the category that this label belongs to

    'categoryId'  , # The ID of this category. Used to create ground truth images
                    # on category level.

    'hasInstances', # Whether this label distinguishes between single instances or not

    'ignoreInEval', # Whether pixels having this class as ground truth label are ignored
                    # during evaluations or not

    'color'       , # The color of this label
    ] )

labels = [
    #       name                     id    trainId   category            catId     hasInstances   ignoreInEval   color
    Label(  'unlabeled'            ,  0 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'ego vehicle'          ,  1 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'rectification border' ,  2 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'out of roi'           ,  3 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'static'               ,  4 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'dynamic'              ,  5 ,      255 , 'void'            , 0       , False        , True         , (111, 74,  0) ),
    Label(  'ground'               ,  6 ,      255 , 'void'            , 0       , False        , True         , ( 81,  0, 81) ),
    Label(  'road'                 ,  7 ,        0 , 'flat'            , 1       , False        , False        , (128, 64,128) ),
    Label(  'sidewalk'             ,  8 ,        1 , 'flat'            , 1       , False        , False        , (244, 35,232) ),
    Label(  'parking'              ,  9 ,      255 , 'flat'            , 1       , False        , True         , (250,170,160) ),
    Label(  'rail track'           , 10 ,      255 , 'flat'            , 1       , False        , True         , (230,150,140) ),
    Label(  'building'             , 11 ,        2 , 'construction'    , 2       , False        , False        , ( 70, 70, 70) ),
    Label(  'wall'                 , 12 ,        3 , 'construction'    , 2       , False        , False        , (102,102,156) ),
    Label(  'fence'                , 13 ,        4 , 'construction'    , 2       , False        , False        , (190,153,153) ),
    Label(  'guard rail'           , 14 ,      255 , 'construction'    , 2       , False        , True         , (180,165,180) ),
    Label(  'bridge'               , 15 ,      255 , 'construction'    , 2       , False        , True         , (150,100,100) ),
    Label(  'tunnel'               , 16 ,      255 , 'construction'    , 2       , False        , True         , (150,120, 90) ),
    Label(  'pole'                 , 17 ,        5 , 'object'          , 3       , False        , False        , (153,153,153) ),
    Label(  'polegroup'            , 18 ,      255 , 'object'          , 3       , False        , True         , (153,153,153) ),
    Label(  'traffic light'        , 19 ,        6 , 'object'          , 3       , False        , False        , (250,170, 30) ),
    Label(  'traffic sign'         , 20 ,        7 , 'object'          , 3       , False        , False        , (220,220,  0) ),
    Label(  'vegetation'           , 21 ,        8 , 'nature'          , 4       , False        , False        , (107,142, 35) ),
    Label(  'terrain'              , 22 ,        9 , 'nature'          , 4       , False        , False        , (152,251,152) ),
    Label(  'sky'                  , 23 ,       10 , 'sky'             , 5       , False        , False        , ( 70,130,180) ),
    Label(  'person'               , 24 ,       11 , 'human'           , 6       , True         , False        , (220, 20, 60) ),
    Label(  'rider'                , 25 ,       12 , 'human'           , 6       , True         , False        , (255,  0,  0) ),
    Label(  'car'                  , 26 ,       13 , 'vehicle'         , 7       , True         , False        , (  0,  0,142) ),
    Label(  'truck'                , 27 ,       14 , 'vehicle'         , 7       , True         , False        , (  0,  0, 70) ),
    Label(  'bus'                  , 28 ,       15 , 'vehicle'         , 7       , True         , False        , (  0, 60,100) ),
    Label(  'caravan'              , 29 ,      255 , 'vehicle'         , 7       , True         , True         , (  0,  0, 90) ),
    Label(  'trailer'              , 30 ,      255 , 'vehicle'         , 7       , True         , True         , (  0,  0,110) ),
    Label(  'train'                , 31 ,       16 , 'vehicle'         , 7       , True         , False        , (  0, 80,100) ),
    Label(  'motorcycle'           , 32 ,       17 , 'vehicle'         , 7       , True         , False        , (  0,  0,230) ),
    Label(  'bicycle'              , 33 ,       18 , 'vehicle'         , 7       , True         , False        , (119, 11, 32) ),
    Label(  'license plate'        , -1 ,       -1 , 'vehicle'         , 7       , False        , True         , (  0,  0,142) ),
]

def browser(path):
    if not os.path.isdir(path):
        raise Exception('Illegal directory: {}'.format(path))
    for _, _, files in os.walk(path):
        return files

def change_label(inputs):
    # ignore license plate
    for i in range(len(labels)-1):
        inputs[inputs == labels[i].id] = labels[i].trainId
    # modify 255 to 19 for training
    inputs[inputs == 255] = 19

    return inputs
######################### STEP 1 ###############################################
def process_gt():
    # root = '/home/cvlab/segdata/CityScape/'
    gt_origin = 'gt/gtFine/'
    gt_new = 'gt/gtNew/'
    # test provides no label images
    top_folders = ['train', 'val']

    # reading subfolders in original folder
    for i in range(2):
        origin_root = os.path.join(gt_origin, top_folders[i])
        new_root = os.path.join(gt_new, top_folders[i])
        subfolders = os.listdir(origin_root)
        subfolders = [item for item in subfolders if os.path.isdir(os.path.join(origin_root, item))]
        # traverse imgs in subfolders
        for fo in subfolders:
            bottom_folder = os.path.join(origin_root, fo)
            new_folder = os.path.join(new_root, fo)

            # if dirs exists
            if not os.path.exists(new_folder):
                os.makedirs(new_folder)
            raw_input('Processing Images in {}'.format(bottom_folder))

            # get image names
            names = browser(bottom_folder)
            names = filter(lambda p: 'labelIds' in p, names)
            # change labels based on label defined above
            for name in names:
                origin_labelpath = os.path.join(bottom_folder, name)
                new_labelpath = os.path.join(new_folder, name)

                origin_label = cv2.imread(origin_labelpath)
                modified_label = change_label(origin_label[..., 0])
                cv2.imwrite(new_labelpath, modified_label)

                print('{} finished!'.format(origin_labelpath))
############################## STEP 2 ##########################################
def divide_img(rgb_savedpath, gt_savedpath, rgb_path, gt_path, rgb_name, gt_name):
    # original size 1024 * 2048, divided into 12 (3*4) parts
    rgb_img = cv2.imread(os.path.join(rgb_path, rgb_name))
    gt_img = cv2.imread(os.path.join(gt_path, gt_name))

    for i in range(3):
        for j in range(4):
            top_left = [i*112, j*416]
            bottom_right = map(lambda a: a+800, top_left)
            rgb_croped = rgb_img[top_left[0]:bottom_right[0], top_left[1]:bottom_right[1]]
            gt_croped = gt_img[top_left[0]:bottom_right[0], top_left[1]:bottom_right[1]]

            postfix = '_{}_{}'.format(i+1,j+1)
            rgb_croped_name = rgb_savedpath + '/' + rgb_name[:-4] + postfix + '.png'
            gt_croped_name = gt_savedpath + '/' + gt_name[:-4] + postfix + '.png'

            cv2.imwrite(rgb_croped_name, rgb_croped)
            cv2.imwrite(gt_croped_name, gt_croped[:,:,0])

            print rgb_croped_name
            # cv2.imshow('rgb_crop', rgb_croped)
            # print rgb_croped.shape
            # cv2.waitKey(500)

def crop_dataset():
    crop_root = ['crop/train/gt/', 'crop/train/leftimg/']
    leftimg = 'leftimg/leftImg8bit/train/'
    gt = 'gt/gtNew/train/'
    subfolders = os.listdir(leftimg)
    for fo in subfolders:
        # make directory in crop_root
        for item in crop_root:
            if not os.path.exists(os.path.join(item, fo)):
                os.makedirs(os.path.join(item, fo))

        # get files' names
        rgbfiles = browser(os.path.join(leftimg, fo))
        # get rgb and label path
        rgb_savedpath = os.path.join(crop_root[1], fo)
        gt_savedpath = os.path.join(crop_root[0], fo)

        for name in rgbfiles:
            labelname = name.replace('leftImg8bit', 'gtFine_labelIds')
            # reading imgs in below path
            rgb_path = os.path.join(leftimg, fo)
            gt_path = os.path.join(gt, fo)
            # crop rgb and groundtruth into 12 parts and save them respectively
            divide_img(rgb_savedpath, gt_savedpath, rgb_path, gt_path, name, labelname)

            print ('Folder: {0}/({1})\t'
                   'Image: {2}/({3})'.format(subfolders.index(fo), len(subfolders)-1,
                   rgbfiles.index(name), len(rgbfiles)-1))

    print 'Croping finished!'
################################ STEP 3 ########################################
def make_text():
    train_root = ['crop/train/gt/', 'crop/train/leftimg/']
    val_root = ['gt/gtNew/val/', 'leftimg/leftImg8bit/val/']

    train_folders = os.listdir(train_root[1])
    val_folders = os.listdir(val_root[1])

    with open('train.txt', 'w') as f:
        print 'Begin to write train.txt'
        for fo in train_folders:
            path = os.path.join(train_root[1], fo)
            rgbnames = browser(path)
            for name in rgbnames:
                gtname = name.replace('leftImg8bit', 'gtFine_labelIds')
                rgb_abs = path + '/' + name
                gt_abs = train_root[0] + fo + '/' + gtname

                f.write(rgb_abs + ' ' + gt_abs + '\n')

                print ('{}/({})\t{}/({})'.format(train_folders.index(fo), len(train_folders)-1,
                                                rgbnames.index(name), len(rgbnames)-1))
    with open('val.txt', 'w') as f:
        print 'Begin to write val.txt'
        for fo in val_folders:
            path = os.path.join(val_root[1], fo)
            rgbnames = browser(path)
            for name in rgbnames:
                gtname = name.replace('leftImg8bit', 'gtFine_labelIds')
                rgb_abs = path + '/' + name
                gt_abs = val_root[0] + fo + '/' + gtname

                f.write(rgb_abs + ' ' + gt_abs + '\n')
                print ('{}/({})\t{}/({})'.format(val_folders.index(fo), len(val_folders)-1,
                                                rgbnames.index(name), len(rgbnames)-1))
if __name__ == '__main__':
    #process_gt()
    #crop_dataset()
    make_text()
