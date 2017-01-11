import numpy as np
import sys
import pickle

caffe-root = '/home/chenqi/caffe-me/'
sys.insert(0, caffe-root + 'python')
import caffe

def ShowNetParam(proto, model):
    #caffe.set_mode_cpu()

    caffe.set_device(1)
    caffe.set_mode_gpu()

    net  = caffe.Net(proto, model, caffe.TEST)

    for k, v in net.params.items():
        print (k, v[0].data.shape)

def CopyWeights(source_solver, target_solver):
    # FCN Layer Name (owing weights):
    origin_layer_list = ['conv1_1', 'conv1_2', 'conv2_1', 'conv2_2', 'conv3_1', \
    'conv3_2', 'conv3_3', 'conv4_1', 'conv4_2', 'conv4_3', 'conv5_1', 'conv5_2', \
    'conv5_3', 'fc6', 'fc7', 'score_fr', 'upscore2', 'score_pool4', 'upscore_pool4', \
    'score_pool3', 'upscore8']
    # solver_source: owned solver.prototxt
    # solver_target: need to be trained net
    solver_source = caffe.get_solver(source_solver)
    solver_target = caffe.get_solver(target_solver)

    # same name of some Layer, deploy.caffemodel is accessed
    solver_source.net.copy_from('deploy.caffemodel')

    target_layer_list = ['conv1_1_v', 'conv1_2_v', 'conv2_1_v', 'conv2_2_v', 'conv3_1_v', \
    'conv3_2_v', 'conv3_3_v', 'conv4_1_v', 'conv4_2_v', 'conv4_3_v', 'conv5_1_v', 'conv5_2_v', \
    'conv5_3_v', 'fc6_v', 'fc7_v', 'score_fr_v', 'upscore2_v', 'score_pool4_v', 'upscore_pool4_v', \
    'score_pool3_v', 'upscore8_v']

    for i in range(len(origin_layer_list)):
        old = origin_layer_list[i]
        new = target_layer_list[i]

        solver_target.net.params[new][0].data[:] = np.copy(solver_source.net.params[old][0].data[:])
        solver_target.net.params[new][1].data[:] = np.copy(solver_source.net.params[old][1].data[:])
    # start training
    solver_target.step(10000)

def pickingmodel(proto, model):
    dict_weights = {}
    # FCN Layer Name (owing weights):
    origin_layer_list = ['conv1_1', 'conv1_2', 'conv2_1', 'conv2_2', 'conv3_1', \
    'conv3_2', 'conv3_3', 'conv4_1', 'conv4_2', 'conv4_3', 'conv5_1', 'conv5_2', \
    'conv5_3', 'fc6', 'fc7', 'score_fr', 'upscore2', 'score_pool4', 'upscore_pool4', \
    'score_pool3', 'upscore8']
    # solver.prototxt
    solver = caffe.get_solver(proto)
    solver.net.copy_from(model)

    for layer_name in origin_layer_list:
        weights, bias = np.copy(solver.net.params[layer_name][0].data[:]), \
        np.copy(solver.net.params[layer_name][1].data[:])

        dict_weights[layer_name] = (weights, bias)

    # pickling weights into ***.pkl file
    pickle.dump(dict_weights, open('weights.pkl', 'w'), True)
    print 'dumping success.'
