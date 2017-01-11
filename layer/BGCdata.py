import numpy as np
import random
import os
from PIL import Image
os.path.insert(0, '/home/chenqi/caffe-me/python')
import caffe

class BGCDataLayer(caffe.Layer):
    """
    Reading different numbers of data from several .h5 files with a fix ratio
    """

    def setup(self, bottom, top):
        params = eval(self.param_str)
        self.split = 'train'
        self.dataT_dir = params['dataT_dir']
        self.dataV_dir = params['dataV_dir']
        self.dataT_propotion = int(params['Tbatch'])
        self.dataV_propotion = int(params['Vbatch'])
        self.idxT = []
        self.idxV = []
        self.random = params.get('randomize', True)

        split_fT = '{}/{}.txt'.format(self.dataT_dir, self.split)
        split_fV = '{}/{}.txt'.format(self.dataV_dir, self.split)

        self.indicesT = open(split_fT, 'r').read().splitlines()
        self.indicesV = open(split_fV, 'r').read().splitlines()

        if len(top) != 2:
            raise Exception("Need to define two tops: data and label.")
        if len(bottom) != 0:
            raise Exception("Do not define a bottom.")

        if self.random:
            self.idxT = random.sample(self.indicesT, self.dataT_propotion)
            self.idxV = random.sample(self.indicesV, self.dataV_propotion)

    def reshape(self, bottom, top):
        self.data = self.load_image()
        self.label = self.load_label()

        # reshape tops
        # since real domain has different size with synthetic domain
        top[0].reshape(self.dataT_propotion + self.dataV_propotion, 3, 360, 480)
        top[1].reshape(self.dataT_propotion + self.dataV_propotion, 1, 360, 480)

    def forward(self, bottom, top):
        """
        Load data
        """
        top[0].data[...] = self.data
        top[1].data[...] = self.label

        # pick next batch
        if self.random:
            self.idxT = random.sample(self.indicesT, self.dataT_propotion)
            self.idxV = random.sample(self.indicesV, self.dataV_propotion)


    def backward(self, bottom, top):
        """
        No back propagate
        """
        pass

    def readresizeimage(imagepath):
        '''
        Load input image and preprocess for Caffe；
        - resize to 360 * 480 to fix CamVid dataset
        - cast to float
        - switch channels RGB -> BGR
        - no substract mean
        - transpose to channel * height * width order
        '''
        im = Image.open(imagepath).resize((480,360))
        in_ = np.array(im, dtype = np.float32)
        in_ = in_[:,:,:,::-1]
        in_ = in_.transpose((2,0,1))
        return in_

    def load_image(self):
        """
        Load images from true dataset and synthetic dataset,
        Each part has different number of images, with a fixed ratio
        """
        count = 0
        images = np.zeros((self.dataT_propotion + self.dataV_propotion, 3, 360, 480))
        # load true dataset images
        for item in self.idxT:
            path = '{}/train/{}'.format(self.dataT_dir, item)
            img = readresizeimage(path)
            images[count,:,:,:] = img
            count += 1
        # load synthetic dataset images
        for item in self.idxV:
            path = '{}/RGB/{}'.format(self.dataV_dir, item)
            img = readresizeimage(path)
            images[count,:,:,:] = img
            count += 1
        return images

    def swaplabel(label, index1, index2):
        # define 100 is not any class
        label[label == index1] =100
        label[label == index2] = index1
        label[label == 100] = index2

        return label

    def resizenp(nparray):
        nparray = nparray.astype(np.uint8)
        im = Image.fromarray(nparray)
        im_resized = im.resize((480,360))

        return np.array(im_resized)

    def readNumpyFromText(path):
        f = open(path, 'r')
        out = np.zeros((720,960), dtype = np.uint8)
        i = 0
        for line in f.readlines():
            perline = line.split(' ')
            out[i,:] = np.array(perline)
            i += 1
        out = resizenp(out)

        print np.amax(out)
        raw_input()
        # making class void 255
        #out -= 1
        # swap label value to fix camvid dataset
        out = swaplabel(out, 2, 6)
        out = swaplabel(out, 7, 8)
        out = swaplabel(out, 7, 6)
        out = swaplabel(out, 7, 4)
        out = swaplabel(out, 3, 4)

        return out

    def load_label(self):
        """
        Load labels into blobs
        Ground-truth in true dataset has PNG format
        Ground-truth in synthetic dataset has TXT format and different label with
        true dataset, which need to be swaped.
        """
        count = 0
        labels = np.zeros((self.dataT_propotion + self.dataV_propotion, 1, 360, 480))
        # load true dataset labels, ignore label 11
        for item in self.idxT:
            path = '{}/trainlabel/{}'.format(self.dataT_dir, item)
            label = np.array(Image.open(path).resize(480,360))
            labels[count, :,:,:] = label
            count += 1

        # load synthetic dataset images, ignore label 0 and -1
        for item in self.idxV:
            path = '{}/GTTXT/{}.txt'.format(self.dataV_dir, item[:-4])
            label = readNumpyFromText(path)
            labels[count,:,:,:] = label
            count += 1

        return labels
