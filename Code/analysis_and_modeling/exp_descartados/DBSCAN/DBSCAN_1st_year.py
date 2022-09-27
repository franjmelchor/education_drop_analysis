#!/usr/bin/env python
# coding: utf-8

# In[1]:


import random

import numpy as np
import pandas as pd
import sklearn
from sklearn import cluster                   # Algoritmos de clustering.
from sklearn import datasets                  # Crear datasets.
from sklearn import manifold                  # Algoritmos de reduccion de dimensionalidad.
from sklearn import decomposition             # Módulo de reducción de dimensionalidad.
from sklearn.utils import check_random_state  # Gestión de números aleatorios.

# Clustering jerárquico y dendrograma.
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.neighbors import NearestNeighbors

# UMAP para reducción de dimensionalidad.
import umap

# Visualizacion.
import matplotlib
import matplotlib.pyplot as plt
from sklearn.cluster import AffinityPropagation
import sys 
import os
sys.path.append(os.path.abspath('/home/fran/Escritorio/i3uex/education_drop_clustering/Code/analysis_and_modeling'))
import dunn_index
from sklearn import metrics



# In[2]:
def main():

    analys_personal_data = pd.read_csv('../../../Data/For_analysis_and_modeling/2nd_quadrimester/analys_personal_data.csv',sep='|')


    # In[3]:


    analys_personal_data.head()


    # In[4]:


    def le_dataset(dset, le_cols, cat_cols):
        from sklearn import preprocessing
        for col in cat_cols:
            le = preprocessing.LabelEncoder()
            le.fit(dset[col].cat.categories)
            le_cols.append(le)
            dset[col] = le.transform(dset[col])


    # In[5]:


    def inverse_le_dataset(dset, le_cols, cat_cols):
        from sklearn import preprocessing
        i = 0
        for col in cat_cols:
            le = le_cols[i]
            from sklearn import preprocessing
            dset[col] = le.inverse_transform(dset[col])
            i +=1


    # In[6]:


    def get_dunn_index(data, labels):
        from sklearn.metrics.pairwise import euclidean_distances
        distances = euclidean_distances(analys_personal_data_clust)
        return dunn_index.dunn(labels,distances)



    # In[7]:


    for col in analys_personal_data.columns:
        if 'object' in str(analys_personal_data.dtypes[col]):
            analys_personal_data[col] = analys_personal_data[col].astype('category')

    le_cols = []
    cat_cols = analys_personal_data.select_dtypes('category').columns
    analys_personal_data_model = analys_personal_data.copy()
    le_dataset(analys_personal_data_model,le_cols,cat_cols)
    analys_personal_data_model.head()


    # In[8]:


    analys_personal_data_model.drop(['expediente','cod_plan'],axis=1,inplace=True)


    # In[9]:


    import plotly as py
    import plotly.express as px
    neigh = NearestNeighbors(n_neighbors=16)
    nbrs = neigh.fit(analys_personal_data_model)
    distances, indices = nbrs.kneighbors(analys_personal_data_model)
    distances = np.sort(distances, axis=0)
    distances = distances[:,1]
    px.line(distances)


    # In[10]:


    eps_arr = list(range(12,20,1))
    min_samples_arr = list(range(14, 21, 1))
    max_metric_eps = -100
    best_epsilon_min_samples = -100,-100
    for eps in eps_arr:
        max_metric_min_samples = -100
        best_min_samples = -100
        for min_samples in min_samples_arr:
            dbscan = cluster.DBSCAN(eps=eps, min_samples=min_samples)
            dbscan.fit(analys_personal_data_model)
            labels = dbscan.labels_
            analys_personal_data_clust = analys_personal_data_model
            analys_personal_data_clust['labels'] = labels
            metric = metrics.silhouette_score(analys_personal_data_clust, analys_personal_data_clust['labels'])
            if metric > max_metric_min_samples:
                max_metric_min_samples = metric
                best_min_samples = min_samples
        if max_metric_min_samples > max_metric_eps:
            max_metric_eps = max_metric_min_samples
            best_epsilon_min_samples = eps,best_min_samples




    # In[11]:


    best_epsilon_min_samples


    # In[12]:


    max_metric_eps


    # In[13]:


    metrics.calinski_harabasz_score(analys_personal_data_clust, analys_personal_data_clust['labels'])


    # In[14]:


    metrics.davies_bouldin_score(analys_personal_data_clust, analys_personal_data_clust['labels'])


    # In[15]:


    get_dunn_index(analys_personal_data_clust,analys_personal_data_clust['labels'])


    # In[17]:


    analys_personal_data['labels'] = analys_personal_data_clust['labels'].astype('category')


    # In[ ]:


    from apitep_utils.report import Report
    for label in analys_personal_data['labels'].cat.categories:
        dset = analys_personal_data[analys_personal_data['labels'] == label]
        report = Report()
        report.generate_advanced(dset,'DBSCAN_1st_year_'+str(label),sys.path[0]+ '/'+ str(label))


if __name__ == "__main__":
    main()