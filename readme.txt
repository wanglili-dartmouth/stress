#Embedding Node Structural Role Identity Using Stress Majorization





##Requirements:
env.yml

##How to use:
Our model				stress\models\sm2vec.py
vis_barbell.py   		embedding visualization on barbell graph
vis_karate.py			embedding visualization on karate network
evaluate_synthetic.py   node clustering experiments on synthetic graphs (default: 'varied' setting)
evaluate_brazil.py		node classification experiments on the brazil air-traffic networks
evaluate_europe.py		node classification experiments on the europe air-traffic networks
evaluate_usa.py			node classification experiments on the usa air-traffic networks

##Code reference:

This code is built upon the code of the following paper:

Claire Donnat, Marinka Zitnik, David Hallac, and Jure Leskovec. "Learning Structural Node Embeddings via Diffusion Wavelets." KDD 2018.     https://github.com/snap-stanford/graphwave


Ribeiro, Leonardo FR, Pedro HP Saverese, and Daniel R. Figueiredo. "struc2vec: Learning node representations from structural identity." KDD 2017.   https://github.com/shenweichen/GraphEmbedding




## Citation

If you find this useful, please use the following citation
```
@inproceedings{wang2021,
  title={Embedding Node Structural Role Identity Using Stress Majorization},
  author={Wang, Lili and Huang, Chenghan and Lu, Ying and Ma, Weicheng and Vosoughi, Soroush},
  booktitle={Proceedings of the 30th ACM International Conference on Information \& Knowledge Management},

}

