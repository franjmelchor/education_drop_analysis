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


def main():

    # In[2]:


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
        distances = euclidean_distances(data)
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


    analys_personal_data


    # In[9]:


    analys_personal_data_model.drop(['expediente','cod_plan'],axis=1,inplace=True)


    # In[10]:


    from sklearn import cluster
    K = range(1, 10)
    sse = []
    for k in K:
        kmeans = cluster.KMeans(n_clusters=k)
        kmeans.fit(analys_personal_data_model)
        sse.append(kmeans.inertia_)


    # In[11]:


    import matplotlib.pyplot as plt
    plt.plot(K, sse, marker='o')
    plt.xlabel('Number of clusters')
    plt.ylabel('Distortion')
    plt.show()


    # In[12]:


    # Silhouette Score for K means
    # Import ElbowVisualizer
    from sklearn.cluster import KMeans
    from yellowbrick.cluster import KElbowVisualizer
    model = KMeans()
    # k is range of number of clusters.
    visualizer = KElbowVisualizer(model, k=(2,30),metric='silhouette', timings= True)
    visualizer.fit(analys_personal_data_model)        # Fit the data to the visualizer
    visualizer.show()        # Finalize and render the figure


    # In[13]:


    kmeans = cluster.KMeans(n_clusters=2)
    kmeans.fit(analys_personal_data_model)
    analys_personal_data_clust = analys_personal_data_model.copy()
    analys_personal_data_clust['labels'] = kmeans.predict(analys_personal_data_model)
    analys_personal_data['labels'] = analys_personal_data_clust['labels']
    analys_personal_data['labels'] = analys_personal_data_clust['labels'].astype('category')


    # In[14]:


    metrics.silhouette_score(analys_personal_data_clust, analys_personal_data_clust['labels'])


    # In[15]:


    metrics.calinski_harabasz_score(analys_personal_data_clust, analys_personal_data_clust['labels'])


    # In[16]:


    metrics.davies_bouldin_score(analys_personal_data_clust, analys_personal_data_clust['labels'])


    # In[17]:


    get_dunn_index(analys_personal_data_clust,analys_personal_data_clust['labels'])


    # In[18]:


    # from apitep_utils.report import Report
    # for label in analys_personal_data['labels'].cat.categories:
    #     dset = analys_personal_data[analys_personal_data['labels'] == label]
    #     report = Report()
    #     report.generate_advanced(dset,'KMeans_1st_year_2_cl_'+str(label),sys.path[0]+ '/'+ str(label))


    # In[19]:


    kmeans = cluster.KMeans(n_clusters=3)
    kmeans.fit(analys_personal_data_model)
    analys_personal_data_clust = analys_personal_data_model
    analys_personal_data_clust['labels'] = kmeans.predict(analys_personal_data_model)
    analys_personal_data['labels'] = analys_personal_data_clust['labels'].astype('category')


    # In[20]:


    metrics.silhouette_score(analys_personal_data_clust, analys_personal_data_clust['labels'])


    # In[21]:


    metrics.calinski_harabasz_score(analys_personal_data_clust, analys_personal_data_clust['labels'])


    # In[22]:


    metrics.davies_bouldin_score(analys_personal_data_clust, analys_personal_data_clust['labels'])


    # In[23]:


    get_dunn_index(analys_personal_data_clust,analys_personal_data_clust['labels'])


    # In[24]:


    # from apitep_utils.report import Report
    # for label in analys_personal_data['labels'].cat.categories:
    #     dset = analys_personal_data[analys_personal_data['labels'] == label]
    #     report = Report()
    #     report.generate_advanced(dset,'KMeans_1st_year_3_cl_'+str(label),sys.path[0]+ '/'+ str(label))


    # In[25]:


    import umap
    umap = umap.UMAP().fit(analys_personal_data_model)
    analys_personal_umap = umap.transform(analys_personal_data_model)
    analys_personal_umap = pd.DataFrame(data=analys_personal_umap, columns=['independent_feature_1', 'independent_feature_2'])


    # In[26]:


    K = range(1, 10)
    sse = []
    for k in K:
        kmeans = cluster.KMeans(n_clusters=k)
        kmeans.fit(analys_personal_umap)
        sse.append(kmeans.inertia_)


    # In[27]:


    plt.plot(K, sse, marker='o')
    plt.xlabel('Number of clusters')
    plt.ylabel('Distortion')
    plt.show()


    # In[28]:


    model = KMeans()
    # k is range of number of clusters.
    visualizer = KElbowVisualizer(model, k=(2,30),metric='silhouette', timings= True)
    visualizer.fit(analys_personal_umap)        # Fit the data to the visualizer
    visualizer.show()


    # In[30]:


    kmeans = cluster.KMeans(n_clusters=3)
    kmeans.fit(analys_personal_umap)
    analys_personal_data_clust = analys_personal_umap.copy()
    analys_personal_data_clust['labels'] = kmeans.predict(analys_personal_umap)
    analys_personal_data['labels'] = analys_personal_data_clust['labels'].astype('category')


    # In[31]:


    import plotly.express as px
    px.scatter(analys_personal_data_clust, x="independent_feature_1", y="independent_feature_2", color='labels')


    # In[ ]:


    from apitep_utils.report import Report
    for label in analys_personal_data['labels'].cat.categories:
        dset = analys_personal_data[analys_personal_data['labels'] == label]
        report = Report()
        report.generate_advanced(dset,'KMeans_1st_year_10_cl_'+str(label),sys.path[0]+ '/'+ str(label))

if __name__ == "__main__":
    main()