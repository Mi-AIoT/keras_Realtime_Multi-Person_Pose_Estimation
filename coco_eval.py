#!/usr/bin/env python
# coding: utf-8

# This is actually dremovd@github code for calculating coco metric.
import sys
import pandas as pd
import os
import glob

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
    gpu = 1
    if gpu == None:
        caffe.set_mode_cpu()
    else:
        caffe.set_device(int(gpu))
        caffe.set_mode_gpu()
    model = caffe.Net(model_params['deployFile'], model_params['caffemodel'], caffe.TEST)
else:
    # Create keras model and load weights
    model = get_testing_model()
    weights_path = "model/keras/model.h5" # orginal weights converted from caffe
    model.load_weights(weights_path)
eval_result_original = validation(model, dump_name = 'original')
joblib.dump(eval_result_original, 'metrics-raw-original.dump')

raw_eval_list = glob.glob('metrics-raw*.dump')

for raw_eval in raw_eval_list:
    eval_result = joblib.load(raw_eval)
    print("\n" + raw_eval)
    eval_result.summarize()
    scores = per_image_scores(eval_result)
    scores.to_csv('%s-scores.csv' % raw_eval)
    print("Average per-image score (not coco metric): %.3f" % scores['average'].mean())
