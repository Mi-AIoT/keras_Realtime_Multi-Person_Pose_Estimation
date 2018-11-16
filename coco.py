#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This is actually dremovd@github code for calculating coco metric. 


# In[2]:


import sys
import pandas as pd


# In[3]:


import os

os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES']='0'


# Create keras model and load weights

# In[4]:


print(sys.version)

from keras.models import Model
from model import get_testing_model
model = get_testing_model(38, 19, 6)
print(os.getcwd())
sys.path.append('/DATA/hqfang/ev_project/reference/keras_Realtime_Multi-Person_Pose_Estimation/testing')
print(sys.path)


# In[5]:


import os
from coco_metric import per_image_scores, validation


# In[6]:


training_dir = './training/'
trained_models = [
    'weights'
    #'weights-cpp-lr',
    #'weights-python-last',
]
optimal_epoch_loss = 'val_weight_stage6_L1_loss'


# In[7]:


from sklearn.externals import joblib

for trained_model in trained_models:
    model_dir = os.path.join(training_dir, trained_model)
    training_log = pd.read_csv(os.path.join(model_dir, 'training.csv'))
    min_index = training_log[[optimal_epoch_loss]].idxmin()
    min_epoch, min_loss = training_log.loc[min_index][['epoch', optimal_epoch_loss]].values[0]
    print("Model '%s', optimal loss: %.3f at epoch %d" % (trained_model, min_loss, min_epoch))
    
    epoch_weights_name = os.path.join(model_dir, 'weights.%04d.h5' % min_epoch)
    print(epoch_weights_name)
    model.load_weights(epoch_weights_name)
    eval_result = validation(model, dump_name = trained_model)
    joblib.dump(eval_result, 'metrics-raw-%s.dump' % trained_model)


# In[11]:


weights_path = os.path.join(os.getcwd(), "model/keras/model.h5") # orginal weights converted from caffe
print(weights_path)
model.load_weights(weights_path)
eval_result_original = validation(model, dump_name = 'original')
joblib.dump(eval_result_original, 'metrics-raw-original.dump')


# In[13]:


raw_eval_list = get_ipython().getoutput('ls metrics-raw*.dump')


# In[14]:


from sklearn.externals import joblib

for raw_eval in raw_eval_list:
    eval_result = joblib.load(raw_eval)
    print("\n" + raw_eval)
    eval_result.summarize()
    scores = per_image_scores(eval_result)
    scores.to_csv('%s-scores.csv' % raw_eval)
    print("Average per-image score (not coco metric): %.3f" % scores['average'].mean())


# In[ ]:





# In[ ]:




