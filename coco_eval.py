#!/usr/bin/env python
# coding: utf-8

# This is actually dremovd@github code for calculating coco metric.
import sys
import pandas as pd
import os
import glob
import argparse

def FILE_EXIST(file):
    if not os.path.exists(file):
        raise argparse.ArgumentTypeError("File {} doesn't exist.".format(file))
    else:
        return file

def DIR_EXIST(dir):
    if not os.path.isdir(dir):
        raise argparse.ArgumentTypeError("Directory {} doesn't exist.".format(dir))
    else:
        return dir

parser = argparse.ArgumentParser()
parser.add_argument('--deploy_proto', type=FILE_EXIST, help='Openpose deploy prototxt')
parser.add_argument('--deploy_weights', type=FILE_EXIST, help='Openpose deploy weights')
parser.add_argument('--dataset', type=str, default="val2014", help='dataset used, supported val2014 or val2017')
parser.add_argument('--evallist', type=FILE_EXIST, help='evaluation list, such as caffe_rtpose/image_info_val2014_1k.txt')
parser.add_argument('--gpu', type=int, choices=range(16), help='Specify which GPU device id to train')
parser.add_argument('--fixed_dumpdir', type=FILE_EXIST, help='Path of folder of fixed point dumped out directory, if specified, will parse it instead of running caffe inference')

args = parser.parse_args()

os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
#os.environ['CUDA_VISIBLE_DEVICES']='0'

sys.path.append(os.path.join(os.getcwd(), 'testing'))

print(sys.version)

USE_CAFFE = os.environ.get('USE_CAFFE')

if USE_CAFFE:
    os.environ['GLOG_minloglevel'] = '2'
    import caffe
else:
    from keras.models import Model
    from model import get_testing_model

from coco_metric import per_image_scores, validation
from sklearn.externals import joblib
from config_reader import config_reader

params, model_params = config_reader()

print(os.getcwd())

#training_dir = './training/'
#trained_models = [
#    'weights'
#    #'weights-cpp-lr',
#    #'weights-python-last',
#]
#optimal_epoch_loss = 'val_weight_stage6_L1_loss'
#
#for trained_model in trained_models:
#    model_dir = os.path.join(training_dir, trained_model)
#    training_log = pd.read_csv(os.path.join(model_dir, 'training.csv'))
#    min_index = training_log[[optimal_epoch_loss]].idxmin()
#    min_epoch, min_loss = training_log.loc[min_index][['epoch', optimal_epoch_loss]].values[0]
#    print("Model '%s', optimal loss: %.3f at epoch %d" % (trained_model, min_loss, min_epoch))
#
#    epoch_weights_name = os.path.join(model_dir, 'weights.%04d.h5' % min_epoch)
#    print(epoch_weights_name)
#    model.load_weights(epoch_weights_name)
#    eval_result = validation(model, dump_name = trained_model)
#    joblib.dump(eval_result, 'metrics-raw-%s.dump' % trained_model)

if USE_CAFFE:
    gpu = args.gpu
    if gpu == None:
        caffe.set_mode_cpu()
    else:
        caffe.set_device(int(gpu))
        caffe.set_mode_gpu()
    if args.deploy_proto:
        model_params['deployFile'] = args.deploy_proto
    if args.deploy_weights:
        model_params['caffemodel'] = args.deploy_weights
    model = caffe.Net(model_params['deployFile'], model_params['caffemodel'], caffe.TEST)
    print("Use Caffe to do accuracy testing!")
    print("Prototxt: {}\r\nCaffemodel: {}".format(model_params['deployFile'], model_params['caffemodel']))
else:
    # Create keras model and load weights
    model = get_testing_model()
    weights_path = "model/keras/model.h5" # orginal weights converted from caffe
    model.load_weights(weights_path)
    print("Use keras to do accuracy testing!")
    print("Load model weights {}".format(weights_path))

validation_ids = None
if args.evallist is not None:
    validation_ids = []
    with open(args.evallist, 'r') as eval_file:
        for line in eval_file.readlines():
            splits = line.split()
            if len(splits) > 1:
                validation_ids.append(int(splits[1]))

eval_result_original = validation(model, dump_name = 'original', validation_ids=validation_ids, dataset=args.dataset, fixed_dumpdir=args.fixed_dumpdir)
joblib.dump(eval_result_original, 'metrics-raw-original.dump')

raw_eval_list = glob.glob('metrics-raw*.dump')

for raw_eval in raw_eval_list:
    eval_result = joblib.load(raw_eval)
    print("\n" + raw_eval)
    eval_result.summarize()
    scores = per_image_scores(eval_result)
    scores.to_csv('%s-scores.csv' % raw_eval)
    print("Average per-image score (not coco metric): %.3f" % scores['average'].mean())
