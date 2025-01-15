# Readme

**Note: This python code is not original to N. Alonso and B. Millidge.**

## Description

### ``Mixture-of-PageRanks: Replacing Long-Context with Real-Time, Sparse GraphRAG''[1]

Recent studies applying RAGs to long-context tasks have two core limitations: 1) there is little focus on making RAG pipelines computationally efficient, and 2) such studies have only tested simple QA tasks and their performance on more challenging tasks is unknown. To address this, N. Alonso and B. Millidge developed an algorithm based on PageRank, a graph-based search algorithm that they call mixture-of-PageRanks (MixPR). MixPR uses a mixture of PageRank-based graph search algorithms implemented using sparse matrices for efficient and cheap search for a variety of complex tasks. According to N. Alonso and B. Millidge, MixPR search achieves state-of-the-art results on a wide range of long-context benchmark tasks, and despite being far more computationally efficient, both existing RAG methods, specialised search architectures and long-context LLMsIt is stated to outperform [1].Due to the use of sparse embedding, retriever is very computationally efficient, capable of embedding and searching millions of tokens within seconds, and runs entirely on the CPU [1].

## Operating environment

I ran this code in wsl2

- wsl2
  - Ubuntu 20.04.6 LTS
- python:3.8.10
- numpy:1.22.3
- networkx:3.1
- scikit-learn:1.3.2



## Referenced paper
- https://arxiv.org/abs/2412.06078