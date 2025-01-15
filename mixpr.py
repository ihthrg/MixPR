import numpy as np
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# sample documents
documents = [
    "The cat sits on the mat.",
    "Dogs and cats are common pets.",
    "The mat is soft and comfortable.",
    "Pets like dogs and cats bring joy to people."
]

# query
query = "Cats and dogs are good pets."

# 1. TF-IDF-based embedding generation.
# vectorizer = TfidfVectorizer()
vectorizer = TfidfVectorizer(
    stop_words=None,  # Disable stopword.
    min_df=1,        # Minimum document frequency set to 1
    token_pattern=r'(?u)\b\w+\b'  # Single letters are also recognised as words.
)

tfidf_matrix = vectorizer.fit_transform(documents)

# 2. Building sparse graphs.
similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)
epsilon = 1e-6  # Small constant to avoid disconnected components
similarity_matrix += epsilon
threshold = 0.1
adj_matrix = (similarity_matrix > threshold).astype(float)

# Debug info
print(f"Number of zeros in adjacency matrix: {(adj_matrix == 0).sum()}")
print(f"Matrix shape: {adj_matrix.shape}")

# 3. Create graphs.
graph = nx.from_numpy_array(adj_matrix)
print(f"Number of nodes: {graph.number_of_nodes()}")
print(f"Number of edges: {graph.number_of_edges()}")

# 4. Embedding queries and calculating similarity
query_vec = vectorizer.transform([query])
query_similarity = cosine_similarity(query_vec, tfidf_matrix).flatten()

# 5. Run personalised PageRank.
personalization = {i: max(score, epsilon) for i, score in enumerate(query_similarity)}
pagerank_scores = nx.pagerank(graph, alpha=0.85, personalization=personalization)

# 6. Display of scores
ranked_documents = sorted(pagerank_scores.items(), key=lambda x: x[1], reverse=True)
print("\nRanking of documents:")
for rank, (doc_idx, score) in enumerate(ranked_documents, 1):
    print(f"{rank}. Document {doc_idx}: {documents[doc_idx]} (Score: {score:.4f})")