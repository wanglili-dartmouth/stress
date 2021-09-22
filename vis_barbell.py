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
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.manifold import TSNE
from sklearn.linear_model import LogisticRegression
import seaborn as sns
from sklearn.manifold import MDS
def plot(embeddings):
    X=[]
    Y=[]
    Label=[]
    for i in range(0,30):
        X.append(embeddings[str(i)][0])
        Y.append(embeddings[str(i)][1])
    canvas_height = 15
    canvas_width = 15
    dot_size = 1000
    text_size = 18
    legend_setting = False #“brief” / “full” / False


    sns.set(style="whitegrid")

    # set canvas height & width
    plt.figure(figsize=(canvas_width, canvas_height))


    color_paltette=[(0,34,255),(136,190,70),(189,43,18),(97,165,246),(223,186,36),(135,101,175),(227,227,227)]
    pts_colors=list(range(30))
    for i in range(30):
        if(i<9 or i>20):
            pts_colors[i]="color_1"
        if(i==9 or i==20):
            pts_colors[i]="color_2"
        if(i==10 or i==19):
            pts_colors[i]="color_3"
        if(i==11 or i==18):
            pts_colors[i]="color_4"
        if(i==12 or i==17):
            pts_colors[i]="color_5"
        if(i==13 or i==16):
            pts_colors[i]="color_6"
        if(i==14 or i==15):
            pts_colors[i]="color_7"

    for i in range(7):
        color_paltette[i] = (color_paltette[i][0] / 255, color_paltette[i][1] / 255, color_paltette[i][2] / 255)
        
        
    # reorganize dataset
    draw_dataset = {'x': X,
                    'y': Y, 
                    'label':list(range(1, 30 + 1)),
                    'ptsize': dot_size,
                    "cpaltette": color_paltette,
                    'colors':pts_colors}

    #draw scatterplot points
    ax = sns.scatterplot(x = "x",y = "y", alpha = 1,s = draw_dataset["ptsize"],hue="colors", palette=draw_dataset["cpaltette"], legend = legend_setting, data = draw_dataset)


    return ax
if __name__ == "__main__":
    random.seed(0)
    np.random.seed(0)
    graph = nx.read_weighted_edgelist("data/barbell.edgelist", delimiter=" ", nodetype=None,create_using=nx.Graph())
    nx.set_edge_attributes(graph, name="weight", values={edge: 1
        for edge, weight in nx.get_edge_attributes(graph, name="weight").items()})


    for our_size in [2]:


############################################################################################
        model = SM2Vec(graph.to_directed(), walk_length=10, num_walks=80,workers=8, verbose=40 )
        embeddings=model.get_embeddings(1,6)
        print(embeddings)
        ax=plot(embeddings)
        ax.axis("equal")
        ax.figure.savefig("barbell.pdf",bbox_inches='tight')

