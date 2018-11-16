import sys
sys.path.append('.')
from model import get_testing_model
import numpy as np
import os

CAFFE_LAYERS_DIR = "model/caffe/layers"
KERAS_MODEL_FILE = "model/keras/model.h5"

stages = 6
np_branch1 = 38
np_branch2 = 19
m = get_testing_model(np_branch1, np_branch2, stages)

for layer in m.layers:
    layer_name = layer.name
    if (os.path.exists(os.path.join(CAFFE_LAYERS_DIR, "W_%s.npy" % layer_name))):
        w = np.load(os.path.join(CAFFE_LAYERS_DIR, "W_%s.npy" % layer_name))
        b = np.load(os.path.join(CAFFE_LAYERS_DIR, "b_%s.npy" % layer_name))

        w = np.transpose(w, (2, 3, 1, 0))

        layer_weights = [w, b]
        layer.set_weights(layer_weights)

if os.path.exists(os.path.dirname(KERAS_MODEL_FILE)) == False:
    os.makedirs(os.path.dirname(KERAS_MODEL_FILE))
m.save_weights(KERAS_MODEL_FILE)

print("Done !")
