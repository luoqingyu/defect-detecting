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
print(immatrix.shape)
#array  -800      -  name_list
'''
#800   *961*384



#逐样本均值消减和归一化
immatrix -=np.mean(immatrix,axis=0) #减去均值，使得以0为中心
immatrix /=np.std(immatrix,axis=0)  #归一化
print len(immatrix)
'''

#k-means
#设定聚类数目
num_cluster=80
#构造聚类器
clf=KMeans(n_clusters=num_cluster,init='k-means++')
#聚类
s=clf.fit(immatrix)


print (clf.cluster_centers_)
#0-799
#

#获取每个样本的标签
#index=clf.fit_predict(immatrix)
#获取聚类标签
label_pred=clf.labels_
#获取聚类中心
centroids=clf.cluster_centers_
#获取聚类准则的总和
inertia=clf.inertia_

#相似度衡量
loss=0
for i in range(immatrix.shape[0]):
#    line=instation[i]+","+str(clf.labels_[i])+'\n'
    loss +=np.sqrt(np.sum(np.square(immatrix[i]-clf.cluster_centers_[clf.labels_[i]])))
    #以轮廓系数作为评价标准，均为0.4086837
#    print(metrics.silhouette_score(immatrix,clf.labels_,metric='euclidean'))

#print(np.shape(label_pred)) (800,0)
#保存模型
joblib.dump(clf , './km.pkl')

#载入保存的模型
clf = joblib.load('./km.pkl')
plt.scatter(centroids[:, 0],centroids[: , 1], marker="o")
plt.show()


