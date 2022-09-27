#!/usr/bin/env python
# coding: utf-8

# In[21]:


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


    # In[23]:


    def get_dunn_index(data, labels):
        from sklearn.metrics.pairwise import euclidean_distances
        distances = euclidean_distances(data)
        return dunn_index.dunn(labels,distances)



    # In[6]:


    for col in analys_personal_data.columns:
        if 'object' in str(analys_personal_data.dtypes[col]):
            analys_personal_data[col] = analys_personal_data[col].astype('category')

    le_cols = []
    cat_cols = analys_personal_data.select_dtypes('category').columns
    analys_personal_data_model = analys_personal_data.copy()
    le_dataset(analys_personal_data_model,le_cols,cat_cols)
    analys_personal_data_model.head()


    # In[7]:


    analys_personal_data_model.drop(['expediente','cod_plan'],axis=1,inplace=True)


    # In[8]:


    def print_dendogram(linked):
        plt.figure(figsize=(10, 7))
        plt.title("Customer Dendograms")
        dend = dendrogram(linked, no_labels=True)


    # In[9]:


    linked = linkage(analys_personal_data_model, 'ward')
    print_dendogram(linked)


    # Según el clustering jerárquico hay 3 grupos (cortamos en la creación del primer grupo)

    # In[11]:


    from sklearn.cluster import AgglomerativeClustering


    # In[12]:


    aglom_cluster = AgglomerativeClustering(n_clusters = 3, affinity='euclidean', linkage='ward')
    aglom_cluster.fit_predict(analys_personal_data_model)


    # In[14]:


    analys_personal_data_clust = analys_personal_data_model.copy()
    analys_personal_data_clust['labels'] = aglom_cluster.labels_
    analys_personal_data['labels'] = analys_personal_data_clust['labels']
    analys_personal_data['labels'] = analys_personal_data_clust['labels'].astype('category')


    # In[16]:


    from sklearn import metrics


    # In[17]:


    metrics.silhouette_score(analys_personal_data_clust, analys_personal_data_clust['labels'])


    # In[18]:


    metrics.calinski_harabasz_score(analys_personal_data_clust, analys_personal_data_clust['labels'])


    # In[19]:


    metrics.davies_bouldin_score(analys_personal_data_clust, analys_personal_data_clust['labels'])


    # In[24]:


    get_dunn_index(analys_personal_data_clust,analys_personal_data_clust['labels'])


    # In[ ]:


    from apitep_utils.report import Report
    for label in analys_personal_data['labels'].cat.categories:
        dset = analys_personal_data[analys_personal_data['labels'] == label]
        report = Report()
        report.generate_advanced(dset,'Hierarchical_1st_year_3_cl_'+str(label),sys.path[0]+ '/'+ str(label))

if __name__ == "__main__":
    main()