# Federated Graph Learning
## Introduction
```math
A graph with a set of multiple relations $R$ can be defined as $$G = (E, R) = \{(s,r,o)\}$$ where $s,o \in E$ and $r \in R$. 
```

**Graph Machine Learning** is to generate low-dimension representations $s,o,r \in R^d$ that can be used to reconstruct the graph as closely as possible. 

**Reconstruction** can be defined in the following terms: Given the loss function $L$, and a scoring function $f$, the task of the graph representation is to minimize $L=f(s,r,o)$ where $\{(s,r,o)\} \in G$, and maximize where $\{(s,r,o)\} \notin G$. 

The are multiple **loss functions** that can be used, for example:
1. **Margin Ranking Loss**: $L = max(0, -y (x_2 - x_1 ) + \gamma)$
2. **Binary Cross Entropy Loss**
3. **Logistic Loss**

There are also multiple **scoring functions** that can be used, with many of them borrow the ideas from NLP, including:
4. $min (s + r - o)$ **TransE**
5. $max <s,r,o>$ **DistMult**
and so on.

In this project, we will be using **TransE** with **Margin Ranking Loss**. We will also implement a negative sampler that will be used to differentiate positives from negatives.

## Setup
The project contain the following pieces:
### Dataset
The dataset to be used is the **countries** dataset.
### ID-Mapping table
For each entity $e \in E$ and $r \in R$, we will be giving it an index to make it easy to lookup for the embedding in the dictionary.
### Negative Sampler
For each triple $\{s,r,o\}$ in the dataset. We will be corrupting the triple by either replace the head $s$ or object $o$ to treat it as a negative sample. Note that a corrupted triple may still be a positive in the original graph.

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
