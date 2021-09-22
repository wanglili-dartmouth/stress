from sklearn.metrics import roc_auc_score
import numpy as np
import random
from stress.classify import read_node_label, Classifier
from stress import *
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.manifold import TSNE
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from karateclub import GraphWave
from sklearn.model_selection import cross_validate
def test(labels,embedding_dict):
    embedding=np.array([(embedding_dict[str(i)]) for i in range(len(labels))])
    print(type(labels))
    print(len(embedding[0]))
    sss = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)
    f1_micros = []
    f1_macros = []
        
  
#########################
    i = 1
    f1_micros = []
    f1_macros = []
    for split_train, split_test in sss.split(embedding, labels):
        model=SVC(gamma='auto')
        model.fit(embedding[split_train], labels[split_train])        
        predictions = model.predict(embedding[split_test])
        f1_micro = f1_score(labels[split_test], predictions, average="micro")
        f1_macro = f1_score(labels[split_test], predictions, average="macro")
        f1_micros.append(f1_micro)
        f1_macros.append(f1_macro)
        i += 1
        print(f1_macros, file = sample)
    print("No feature: ",np.mean(f1_micros), np.mean(f1_macros), file = sample)
    
    #########################33

    return 
def test_grid(labels,embedding_dict):
    

    embedding=np.array([(embedding_dict[str(i)]) for i in range(len(labels))])
    scoring = ['f1_macro', 'f1_micro']
    clf = svm.SVC( C=25,gamma=0.0001, random_state=0)
    scores = cross_validate(clf, embedding, labels, cv=10, scoring=scoring)
    print("10 cross val: ",np.mean(scores['test_f1_macro']), np.mean(scores['test_f1_micro']), file = sample)
    return 
if __name__ == "__main__":
    random.seed(0)
    np.random.seed(0)
    graph = nx.read_weighted_edgelist("data/europe-airports.edgelist", delimiter=" ", nodetype=None,create_using=nx.Graph())
    graph_int = nx.read_weighted_edgelist("data/europe-airports.edgelist", delimiter=" ", nodetype=int,create_using=nx.Graph())
    labels = pd.read_csv('data/labels-europe-airports.txt', index_col=0, sep=" ")
    labels=labels.values

    nx.set_edge_attributes(graph, name="weight", values={edge: 1
        for edge, weight in nx.get_edge_attributes(graph, name="weight").items()})

    sample = open('europe2.out', 'w') 
    model = SM2Vec(graph.to_directed(), walk_length=10, num_walks=80,workers=8, verbose=40)
    embeddings=model.get_embeddings(0.6,1,dim=128)
    test_grid(labels,embeddings)
    sample.flush()
   
    ########################
                