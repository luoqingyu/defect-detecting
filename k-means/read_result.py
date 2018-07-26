# -*- coding: utf-8 -*-
'''
\cut中含有800多张图片，对其进行聚类。
'''
import os
from PIL import Image
from sklearn.cluster import KMeans
from sklearn import metrics
import matplotlib.pyplot as plt
import numpy as np
from pylab import *
import pickle
from scipy import *
from sklearn.externals import joblib
from scipy.cluster.vq import *

#create a list of images and get their scale

path='/home/veritas7/project/clustercv/kmeans01/cut/'  #path='/home/veritas7/project/clustercv/kmeans01/cut/'   #绝对路径
imlist = [os.path.join(path,f) for f in os.listdir(path) if f.endswith('.bmp')]
#
#800个图片地址
imnbr=len(imlist)
print ("The number of images is %d" %imnbr)

#Create matrix to store all flattened images

immatrix = array([array(Image.open(imname)).flatten() for imname in imlist],'f')
#载入保存的模型
clf = joblib.load('./km.pkl')
#print (clf.cluster_centers_)
#print (clf.cluster_centers_)
print(immatrix[1:5,:].shape)
print (clf.predict(immatrix))
#print (clf.labels_)
#print(clf.cluster_centers_[clf.labels_[5]].shape)
