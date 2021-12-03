# Federated Graph Learning
## Introduction
A graph with a set of multiple relations <img src="https://render.githubusercontent.com/render/math?math=R"> can be defined as <img src="https://render.githubusercontent.com/render/math?math=G = (E, R) = \{(s,r,o)\}"> where <img src="https://render.githubusercontent.com/render/math?math=s,o \in E"> and <img src="https://render.githubusercontent.com/render/math?math=r \in R">. 


**Graph Machine Learning** is to generate low-dimension representations <img src="https://render.githubusercontent.com/render/math?math=s,o,r \in R^d"> that can be used to reconstruct the graph as closely as possible. 

**Reconstruction** can be defined in the following terms: Given the loss function <img src="https://render.githubusercontent.com/render/math?math=L">, and a scoring function <img src="https://render.githubusercontent.com/render/math?math=f">, the task of the graph representation is to minimize <img src="https://render.githubusercontent.com/render/math?math=L=f(s,r,o)"> where <img src="https://render.githubusercontent.com/render/math?math=\{(s,r,o)\} \in G">, and maximize where <img src="https://render.githubusercontent.com/render/math?math=\{(s,r,o)\} \notin G">. 

The are multiple **loss functions** that can be used, for example:
1. **Margin Ranking Loss**: <img src="https://render.githubusercontent.com/render/math?math=L = max(0, -y (x_2 - x_1 ) + (\gamma))">
2. **Binary Cross Entropy Loss**
3. **Logistic Loss**

There are also multiple **scoring functions** that can be used, with many of them borrow the ideas from NLP, including:

4. <img src="https://render.githubusercontent.com/render/math?math=min (s + (r) - o)"> **TransE**

5. <img src="https://render.githubusercontent.com/render/math?math=max <s,r,o>"> **DistMult**
and so on.

In this project, we will be using **TransE** with **Margin Ranking Loss**. We will also implement a negative sampler that will be used to differentiate positives from negatives.

## Setup
The project contain the following pieces:
### Dataset
The dataset to be used is the **countries** dataset.
### ID-Mapping table
For each entity <img src="https://render.githubusercontent.com/render/math?math=e \in E"> and <img src="https://render.githubusercontent.com/render/math?math=r \in R">, we will be giving it an index to make it easy to lookup for the embedding in the dictionary.
### Negative Sampler
For each triple <img src="https://render.githubusercontent.com/render/math?math=\{s,r,o\}"> in the dataset. We will be corrupting the triple by either replace the head <img src="https://render.githubusercontent.com/render/math?math=s"> or object <img src="https://render.githubusercontent.com/render/math?math=o"> to treat it as a negative sample. Note that a corrupted triple may still be a positive in the original graph.

## Distributed Graph Learning
### General Idea
Distributed Graph Learning can be easily implemented using **pytorch** and **pool.map**. The idea is simply to create batches by the entity sequence, and deliver different batches to different subprocesses. 
### Detailed Setup
The implementation is described below:![enter image description here](https://github.com/Otamio/FederatedGraphLearning/blob/main/rsc/1.PNG)
## Federated Graph Learning
### General Idea
The difference between federated graph learning and distributed learning is that most of the training will be done on the client side, the job of the master is to collect and average the gradients collected from the clients.
### Detailed Setup
The implementation is described below:
![enter image description here](https://github.com/Otamio/FederatedGraphLearning/blob/main/rsc/2.PNG)
