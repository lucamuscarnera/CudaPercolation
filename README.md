# CudaPercolation
 Simulation of a Percolation, written in CUDA. Objective : construct a tool to analyze the distribution of cluster in function of variation of the order parameter of the system


# Abstract

Percolation is one of the most interesting model proposed in the framework of Statistical Mechanics. From a Mathematical perspective, percolation can be interpreted as a stochastic process indexed by $\mathbb Z^d$ (or $\mathbb N^d$) whose variables are Bernoulli distributed, with a common parameter $p$. Trivially, if $p=0$ then the realizations of the stochastic process will always yield $0$, if $p=1$ we will have a trivially "always on" behaviour and if $p \in (0,1)$ we will have "interesting" behaviours (certainly worth to be simulated!).

In a percolation model, we can construct a function $C : Z^d \times Z^d \rightarrow \{0,1\}$ that describes if two random variables belong to the same cluster, according to the condition

$$
C((i_1,i_2,...,i_d),(i'_1,i'_2,...,i'_d)) = 1 \iff \exists path({\bf i},{\bf i}')
$$

Our objective is to study the statistical properties of clusters by varying the order parameter $p$.
