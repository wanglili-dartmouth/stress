import matplotlib.pyplot as plt
import networkx as nx 
import numpy as np
import pandas as pd
import pickle
import random
import seaborn as sb
import sklearn as sk
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import sys
from sklearn.neighbors import KNeighborsClassifier
from shapes.shapes import *
from sklearn.metrics import pairwise_distances
from shapes.build_graph import *
from math import sqrt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import json
from stress import *

    
width_basis = 40
name_graph='regular'
basis_type = "cycle" 
add_edges=0
list_shapes=[["fan",6]]*8+[["star",6]]*8+[["house"]]*8
sample = open('varied.out', 'w') 

sb.set_style('white')

Total_times=25
ami=0
sil=0
hom=0
comp=0
f1_micro=0
f1_macro=0
random.seed(0)
np.random.seed(0)
for times in range(Total_times):
    G, communities, plugins, role_id = build_structure(width_basis, basis_type, list_shapes, start=0,
                            rdm_basis_plugins =False, add_random_edges=add_edges,
                            plot=False, savefig=False)
    G=nx.relabel_nodes(G, lambda x: str(x))
    print( 'nb of nodes in the graph: ', G.number_of_nodes() )
    print( 'nb of edges in the graph: ', G.number_of_edges()  )

    model = SM2Vec(G.to_directed(), temp_path="./varied/", workers=8, verbose=40 )
    embeddings=model.get_embeddings(1,1,dim=128)
    trans_data=np.array([(embeddings[str(i)]) for i in range(len(role_id))])
    colors = role_id
    nb_clust = len(np.unique(colors))
    km = sk.cluster.AgglomerativeClustering(n_clusters=nb_clust,linkage='single')
    km.fit(trans_data)
    labels_pred = km.labels_

    labels = colors

    ami+=sk.metrics.adjusted_mutual_info_score(colors, labels_pred) / Total_times 
    sil+=sk.metrics.silhouette_score(trans_data,labels_pred, metric='euclidean')  / Total_times
    hom+=sk.metrics.homogeneity_score(colors, labels_pred) / Total_times
    comp+=sk.metrics.completeness_score(colors, labels_pred)/  Total_times

print("SM2Vec:  ", file = sample)
print ('Homogeneity \t Completeness \t AMI \t nb clusters \t Silhouette  \n', file = sample)
print (str(hom)+'\t'+str(comp)+'\t'+str(ami)+'\t'+str(nb_clust)+'\t'+str(sil), file = sample)
sample.flush()