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
    for i in range(1,69):
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

    pts_num = 68
    # add categorical information ()
    pts_colors = list(range(pts_num))

    lnum = [21,19,27,15,23,30,16,33,24,34,10,31,29,28,26,14, 3, 9,32,25, 4, 8, 2,20,13,22, 1,18, 6,11,17, 7, 5,12]
    rnum = [46,61,68,64,47,36,66,51,35,42,58,65,56,45,57,60,38,41,48,44,59,43,39,54,62,49,37,40,50,53,52,55,63,67]


    for i in range(len(lnum)):
        pts_colors[lnum[i] - 1] = "color_" + str(i + 1)
        pts_colors[rnum[i] - 1] = "color_" + str(i + 1)
        

    # set color paltette
    color_paltette_0 = [(103, 99, 222),(194, 154, 223),(64,111,87),(126,128,219),(198,103,62),(59,134,138),(177,105,42),(136,41,191),(111,236,113),(82,130,129),(67,19,62),(229,82,40),(226,72,200),(120,188,197),(134,241,185),(157,116,156),(192,137,224),(103,211,168),(86,118,192),(107,74,28),(113,128,224), (148,176,176), (62,108,119),(176,105,175),(41,96,181),(220,76,203),(152,229,115),(214,51,54),(235,79,237),(142,197,138),(63,52,240),(232,242,79),(95,202,119),(181,195,235)]
    color_paltette=list(range(len(color_paltette_0)))
    for i in range(len(lnum)):
        j = min(lnum[i], rnum[i]) - 1
        color_paltette[j] = color_paltette_0[i]
        
    for i in range(len(lnum)):
        color_paltette[i] = (color_paltette[i][0] / 255, color_paltette[i][1] / 255, color_paltette[i][2] / 255)
        
        
    # reorganize dataset
    draw_dataset = {'x': X,
                    'y': Y, 
                    'label':list(range(1, pts_num + 1)),
                    'ptsize': dot_size,
                    "cpaltette": color_paltette,
                    'colors':pts_colors}

    #draw scatterplot points
    ax = sns.scatterplot(x = "x",y = "y", alpha = 1,s = draw_dataset["ptsize"],hue="colors", palette=draw_dataset["cpaltette"], legend = legend_setting, data = draw_dataset)



    # add text on point circle
    for i in range(pts_num-34):
        ax.text(X[i], Y[i], i+1 ,horizontalalignment='center',verticalalignment='center', size=text_size, color='white', weight='semibold')

        


    return ax
if __name__ == "__main__":
    random.seed(0)
    np.random.seed(0)
    graph = nx.read_weighted_edgelist("data/karate-mirrored2.edgelist", delimiter=" ", nodetype=None,create_using=nx.Graph())
    nx.set_edge_attributes(graph, name="weight", values={edge: 1
        for edge, weight in nx.get_edge_attributes(graph, name="weight").items()})


    for our_size in [2]:


############################################################################################
        model = SM2Vec(graph.to_directed(), walk_length=10, num_walks=80,workers=8, verbose=40 )
        embeddings=model.get_embeddings(1,3)
        print(embeddings)
        ax=plot(embeddings)
        ax.axis("equal")
        ax.figure.savefig("karate.pdf",bbox_inches='tight')

