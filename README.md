# Vision Transformer with Token Merging using Bipartite Soft Merging

This repository contains an implementation of Vision Transformers (ViT) with a token merging mechanism using Bipartite Soft Merging from the paper https://arxiv.org/abs/2210.09461. The objective is to enhance the throughput of Vision Transformers by merging tokens in an adaptive manner. Includes training code.

## Introduction


## Features

- **Vision Transformer (ViT) Implementation**: Based on the original ViT architecture.
- **Bipartite Soft Merging**:merge tokens effectively, reducing computational load.

---
layout: default
mathjax: true
title: Bipartite Soft Matching Algorithm
---

# Bipartite Soft Matching Algorithm

1. Partition the token set 
   \[
   T = \{ t_1, t_2, \dots, t_n \}
   \]
   into two disjoint subsets 
   \[
   A = \{ a_1, a_2, \dots, a_k \} \quad \text{and} \quad B = \{ b_1, b_2, \dots, b_k \},
   \]
   where 
   \[
   k \approx \frac{n}{2}.
   \]

2. Compute the similarity matrix 
   \[
   S \in \mathbb{R}^{k \times k}
   \]
   where each entry 
   \[
   S_{ij} = S(a_i, b_j)
   \]
   represents the similarity between token \(a_i \in A\) and \(b_j \in B\).

3. Select the top \( r \) pairs based on similarity, i.e., find the edges 
   \[
   \{ (a_i, b_j) \}_{i,j}
   \]
   with the largest \( S_{ij} \) values, where \( r \) is the desired number of token mergers.

4. Merge tokens by averaging their feature vectors: for each selected pair \( (a_i, b_j) \), compute the merged token
   \[
   m_{ij} = \frac{a_i + b_j}{2},
   \]
   and replace the pair with \( m_{ij} \).

5. Concatenate the merged tokens back into the final set:
   \[
   T' = \{ m_{ij}, \dots \},
   \]
   where \( T' \) is the set of merged tokens.



## Installation

To get started, clone the repository and install the necessary dependencies:

```bash
git clone https://github.com/Ctrl408/ViT-implementations.git
cd ViT-implementations
```

