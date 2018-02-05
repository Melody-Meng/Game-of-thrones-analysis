#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
import io
import pandas as pd
import numpy as np
import nltk
import pylab as pl
from nltk.stem.snowball import SnowballStemmer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt

from GetData import *

a = GetData

# Get top 100 words from all episodes
data = []
for i in range(1,8):
    f = io.open("sub" + str(i) + ".txt", encoding='utf-8').read()
    stem = a.word_extract(f)
    data.extend(list(stem))

# calculate 50 most frequent words of the season
fre_100 = a.most_100(data)
fre_100_value = [x[0] for x in fre_100]


# get keyword existence in each episode
def CheckExistence(n):
    existence = np.empty(shape=(n-1,100), dtype=np.int32)    
    for i in range (1,n):
        f = io.open("sub" + str(i) + ".txt", encoding='utf-8').read()
        stem = a.word_extract(f)
        word = a.unique(stem)
        exist = []
        for keyword in fre_100_value:
            if keyword in word:
                exist.append(stem.count(keyword))
            else:
                exist.append(0)
        existence[i-1] = exist
    return pd.DataFrame(existence).T

exi = CheckExistence(8)



# K means to center
kmeans = KMeans(n_clusters=6)
kmeans.fit(exi)

labels = kmeans.predict(exi)
centroids = kmeans.cluster_centers_

# PCA to lower dimension
pca = PCA(n_components=2).fit(exi)
pca_2d = pca.transform(exi)
center = pca.transform(centroids)


# plot
plt.rcParams['figure.figsize'] = (16, 9)
plt.style.use('ggplot')

# original scatter plot
fig0 = plt.figure('Scatter Plot')
ax = plt.subplot(111)
ax.scatter(pca_2d[:, 0], pca_2d[:, 1])
plt.show()
fig0.savefig('Scatter Plot.png')

# plot with cluters
fig1 = plt.figure('K-means with 5 clusters')
for i in range(0, pca_2d.shape[0]):
    if kmeans.labels_[i] == 0:
        c1 = plt.scatter(pca_2d[i,0],pca_2d[i,1],c='tomato',marker='^')
    elif kmeans.labels_[i] == 1:
        c2 = plt.scatter(pca_2d[i,0],pca_2d[i,1],c='teal',marker='o')
    elif kmeans.labels_[i] == 2:
        c3 = plt.scatter(pca_2d[i,0],pca_2d[i,1],c='tan',marker='*')
    elif kmeans.labels_[i] == 3:
        c4 = plt.scatter(pca_2d[i,0],pca_2d[i,1],c='wheat',marker='s')
    elif kmeans.labels_[i] == 4:
        c5 = plt.scatter(pca_2d[i,0],pca_2d[i,1],c='orange',marker='+')
    elif kmeans.labels_[i] == 5:
        c6 = plt.scatter(pca_2d[i,0],pca_2d[i,1],c='r',marker='v')

plt.legend([c1, c2, c3, c4, c5,c6],['Cluster 0', 'Cluster 1','Cluster 2','Cluster 3','Cluster 4','Cluster 5'])
plt.scatter(center[:, 0], center[:, 1],
            marker='+', s=110, 
            color='grey', zorder=10,alpha=0.3)
plt.show()
fig1.savefig('K-means with 6 clusters.png')

# plot with stem
fig2 = plt.figure('Clusters with stem')
for i in range(0, pca_2d.shape[0]):
    if kmeans.labels_[i] == 0:
        c1 = plt.scatter(pca_2d[i,0],pca_2d[i,1],c='tomato',marker='^',alpha=0.7)
    elif kmeans.labels_[i] == 1:
        c2 = plt.scatter(pca_2d[i,0],pca_2d[i,1],c='teal',marker='o',alpha=0.7)
    elif kmeans.labels_[i] == 2:
        c3 = plt.scatter(pca_2d[i,0],pca_2d[i,1],c='tan',marker='*',alpha=0.7)
    elif kmeans.labels_[i] == 3:
        c4 = plt.scatter(pca_2d[i,0],pca_2d[i,1],c='wheat',marker='s',alpha=0.7)
    elif kmeans.labels_[i] == 4:
        c5 = plt.scatter(pca_2d[i,0],pca_2d[i,1],c='orange',marker='+',alpha=0.7)
    elif kmeans.labels_[i] == 5:
        c6 = plt.scatter(pca_2d[i,0],pca_2d[i,1],c='r',marker='v')

plt.legend([c1, c2, c3, c4, c5,c6],['Cluster 0', 'Cluster 1','Cluster 2','Cluster 3','Cluster 4','Cluster 5'])

plt.scatter(center[:, 0], center[:, 1],
            marker='+', s=110, 
            color='grey', zorder=10,alpha=0.3)

ax = plt.subplot()           
for i, txt in enumerate(fre_100_value):
    ax.annotate(txt, (pca_2d[:, 0][i],pca_2d[:, 1][i] ), fontsize = 8)
plt.show()
fig2.savefig('Clusters with stem.png')



# Save results to csv
ori_data = pd.DataFrame(np.column_stack((exi, fre_100_value)))
cluster_data = pd.DataFrame(np.column_stack((pca_2d, labels, fre_100_value)))
fre_100 = pd.DataFrame(fre_100)
center = pd.DataFrame(center)

ori_data.to_csv('ori_data.csv',encoding='utf-8', index=False)
cluster_data.to_csv('cluster_data.csv',encoding='utf-8', index=False )
fre_100.to_csv('top_100.csv',encoding='utf-8', index=False )
center.to_csv('center.csv',encoding='utf-8', index=False)


