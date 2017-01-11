import numpy as np
import os

os.path.insert(0, '/home/chenqi/caffe/pyhton')
import caffe

"""Loss Layer in Prototxt File
layer{
type: ’Python'
name: 'loss'
top: 'loss'
bottom： ‘ipx’
bottom: 'ipy'
python_param{
#module的名字，通常是定义Layer的.py文件的文件名，需要在$PYTHONPATH下
module: 'pyloss'
#layer的名字---module中的类名
layer: 'EuclideanLossLayer'
}
loss_weight: 1
}
"""


class EuclideadLossLayer(caffe.layer):
    """docstring for EuclideadLossLayer"""
    def setup(self, bottom, top):
        if len(bottom) != 2:
            raise Exception("Need two inputs to compute distance.")

    def reshape(self, bottom, top):
        if bottom[0].count != bottom[1].count:
            raise Exception("Inputs must have the same dimension.")
        self.diff = np.zeros_like(bottom[0].data, dtype = np.float32)
        top[0].reshape(1)

    def forward(self, bottom, top):
        self.diff[...] = bottom[0].data - bottom[1].data
        top[0].data[...] = np.sum(self.diff**2)/bottom[0].num/2

    def backward(self, top, propagate_down, bottom):
        for i in range(2):
            if not propagate_down[i]:
                continue
            if i == 0:
                sign = 1
            else:
                sign = -1
            bottom[i].diff[...] = sign * self.diff / bottom[i].num
