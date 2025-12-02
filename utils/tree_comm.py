"""
Community detection and hierarchical tree construction module.
Uses FastTreeComm algorithm to cluster graph nodes and create community super-nodes.
"""

import json
import time
import warnings
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Any

import networkx as nx
import numpy as np
import scipy.sparse as sp
import torch
import json_repair
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

from utils import call_llm_api
from utils.logger import logger


warnings.filterwarnings('ignore')

try:
    from config import get_config
except ImportError:
    get_config = None


class FastTreeComm:
    """
    Fast Tree Community Detection class.

    This class implements an algorithm to detect communities in a knowledge graph
    by combining structural information (adjacency) and semantic information (node embeddings).
    It also supports generating summaries for detected communities using an LLM.
    """

    def __init__(self, graph: nx.DiGraph, embedding_model: str = "all-MiniLM-L6-v2", struct_weight: float = 0.3, config: Any = None):
        """
        Initialize the FastTreeComm instance.

        Args:
            graph (nx.DiGraph): The input knowledge graph.
            embedding_model (str): Name of the sentence embedding model to use.
            struct_weight (float): Weight for structural similarity (0 to 1).
                                   (1 - struct_weight) is used for semantic similarity.
            config (Any, optional): Configuration object. If provided, overrides defaults.
        """
        if config is None and get_config is not None:
            try:
                config = get_config()
            except:
                config = None
        self.config = config
        self.graph = graph

        if config:
            embedding_model = embedding_model or config.tree_comm.embedding_model
            struct_weight = struct_weight if struct_weight != 0.3 else config.tree_comm.struct_weight
        
        self.model = SentenceTransformer(embedding_model)
        self.semantic_cache: Dict[str, np.ndarray] = {}
        self.struct_weight = struct_weight
        self.node_list = list(graph.nodes())
        self.node_names = {n: graph.nodes[n]["properties"]["name"] for n in graph.nodes()}
        self.neighbor_cache = {n: set(graph.neighbors(n)) for n in graph.nodes()}
        self.edge_relations = {(u, v): data.get("relation", "related_to") 
                          for u, v, data in graph.edges(data=True)}
        
        self.triple_strings_cache: Dict[str, List[str]] = {}
        self.degree_cache = {n: self.graph.degree(n) for n in self.node_list}

        self.adjacency_sparse = self._build_sparse_adjacency()

        self._precompute_all_triples()
        
        self.llm_client = call_llm_api.LLMCompletionCall()

    def _build_sparse_adjacency(self) -> sp.csr_matrix:
        """
        Build a sparse adjacency matrix for the graph.

        Returns:
            sp.csr_matrix: Sparse adjacency matrix.
        """
        n = len(self.node_list)
        node_to_idx = {node: i for i, node in enumerate(self.node_list)}
        row, col = [], []
        
        for node in self.node_list:
            i = node_to_idx[node]
            for neighbor in self.graph.neighbors(node):
                if neighbor in node_to_idx:
                    j = node_to_idx[neighbor]
                    row.append(i)
                    col.append(j)
        
        data = [1] * len(row)
        return sp.csr_matrix((data, (row, col)), shape=(n, n))

    def _precompute_all_triples(self):
        """Precompute and cache triple strings for all nodes."""
        for node_id in self.node_list:
            self.triple_strings_cache[node_id] = self._get_triple_strings(node_id)
        
        return

    def _get_triple_strings(self, node_id: str) -> List[str]:
        """
        Extract all relationship triples for a node to enhance structural perception.

        Args:
            node_id (str): The ID of the node.

        Returns:
            List[str]: A list of strings representing the triples (e.g., "NodeA relation NodeB").
        """
        if node_id in self.triple_strings_cache:
            return self.triple_strings_cache[node_id]
            
        node_name = self.graph.nodes[node_id]["properties"]["name"]
        triples = []
        
        for neighbor in self.graph.neighbors(node_id):
            rel = self.graph.edges[node_id, neighbor, 0].get("relation", "related_to")
            neighbor_name = self.graph.nodes[neighbor]["properties"]["name"]
            triples.append(f"{node_name} {rel} {neighbor_name}")
            
        result = list(set(triples))
        self.triple_strings_cache[node_id] = result
        return result

    def get_triple_embedding(self, node_id: str) -> np.ndarray:
        """
        Get the embedding for a node based on its triples (context).

        Args:
            node_id (str): The ID of the node.

        Returns:
            np.ndarray: The embedding vector.
        """
        if node_id not in self.semantic_cache:
            triples = self.triple_strings_cache.get(node_id, [])
            text = ", ".join(triples) if triples else self.graph.nodes[node_id]["properties"]["name"]
            self.semantic_cache[node_id] = self.model.encode(text)
        return self.semantic_cache[node_id]
    
    def get_triple_embeddings_batch(self, node_ids: List[str]) -> np.ndarray:
        """
        Get embeddings for a batch of nodes, using optimized batch processing.

        Args:
            node_ids (List[str]): List of node IDs.

        Returns:
            np.ndarray: Array of embedding vectors.
        """
        uncached_ids = [nid for nid in node_ids if nid not in self.semantic_cache]
        
        if uncached_ids:
            texts = []
            for nid in uncached_ids:
                triples = self.triple_strings_cache.get(nid, [])
                text = " ".join(triples) if triples else self.node_names[nid]
                texts.append(text)
            
            with torch.no_grad():
                embeddings = self.model.encode(texts, convert_to_tensor=True, batch_size=128)
                
            for nid, emb in zip(uncached_ids, embeddings):
                self.semantic_cache[nid] = emb.cpu().numpy()
        return np.array([self.semantic_cache[nid] for nid in node_ids])

    def _compute_jaccard_matrix_vectorized(self, level_nodes: List[str]) -> np.ndarray:
        """
        Compute the Jaccard similarity matrix for a subset of nodes based on graph structure.

        Args:
            level_nodes (List[str]): List of node IDs to compute similarity for.

        Returns:
            np.ndarray: Jaccard similarity matrix.
        """
        node_to_idx = {node: i for i, node in enumerate(self.node_list)}
        level_indices = [node_to_idx[node] for node in level_nodes if node in node_to_idx]

        if not level_indices:
            return np.zeros((len(level_nodes), len(level_nodes)))

        sub_adj = self.adjacency_sparse[level_indices][:, level_indices]
        intersection = sub_adj.dot(sub_adj.T).toarray()
        row_sums = np.array(sub_adj.sum(axis=1)).flatten()

        union = row_sums[:, None] + row_sums - intersection
        jaccard_matrix = intersection / (union + 1e-9)
        np.fill_diagonal(jaccard_matrix, 1.0)

        return jaccard_matrix

    def _compute_sim_matrix(self, level_nodes: List[str]) -> np.ndarray:
        """
        Compute the combined similarity matrix (structural + semantic).

        Args:
            level_nodes (List[str]): List of node IDs.

        Returns:
            np.ndarray: Combined similarity matrix.
        """
        start_time = time.time()
        
        node_count = len(level_nodes)
        if node_count <= 1:
            return np.eye(node_count)

        embeddings = self.get_triple_embeddings_batch(level_nodes)
        
        embeddings_normalized = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-9)
        semantic_sim_matrix = np.dot(embeddings_normalized, embeddings_normalized.T)

        structural_sim_matrix = self._compute_jaccard_matrix_vectorized(level_nodes)
        
        sim_matrix = (self.struct_weight * structural_sim_matrix + 
                     (1 - self.struct_weight) * semantic_sim_matrix)
        return sim_matrix

    def _fast_clustering(self, level_nodes: List[str], n_clusters: Optional[int] = None) -> Dict[int, List[str]]:
        """
        Perform fast initial clustering using K-Means on embeddings.

        Args:
            level_nodes (List[str]): List of nodes to cluster.
            n_clusters (Optional[int]): Number of clusters to form. If None, calculated automatically.

        Returns:
            Dict[int, List[str]]: Dictionary mapping cluster IDs to lists of node IDs.
        """
        if len(level_nodes) <= 2:
            return {0: level_nodes}
        
        if n_clusters is None:
            base_clusters = len(level_nodes) // 10
            n_clusters = min(max(2, base_clusters), len(level_nodes) // 2, 200)
        
        embeddings = self.get_triple_embeddings_batch(level_nodes)
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=5)
        cluster_labels = kmeans.fit_predict(embeddings)
        
        clusters = defaultdict(list)
        for node, label in zip(level_nodes, cluster_labels):
            clusters[label].append(node)
        
        return dict(clusters)

    def detect_communities(self, level_nodes: List[str], max_iter: int = 1, merge_threshold: float = 0.5, max_total_communities: Optional[int] = None) -> Dict[int, List[str]]:
        """
        Detect communities within the given set of nodes.

        Uses hierarchical clustering refinement.

        Args:
            level_nodes (List[str]): List of nodes to process.
            max_iter (int): Maximum iterations for refinement.
            merge_threshold (float): Threshold for merging clusters.
            max_total_communities (Optional[int]): Maximum number of communities to produce.

        Returns:
            Dict[int, List[str]]: Detected communities (ID -> Node List).
        """
        if len(level_nodes) <= 1:
            return {0: level_nodes} if level_nodes else {}

        # Read max_total_communities from config if not provided
        if max_total_communities is None:
            if self.config and hasattr(self.config.tree_comm, 'max_total_communities'):
                max_total_communities = self.config.tree_comm.max_total_communities
            else:
                # Default logic: 1/3 of nodes, min 5, max 200
                max_total_communities = min(max(5, len(level_nodes) // 3), 200)

        initial_clusters = self._fast_clustering(level_nodes)
        final_communities = {}
        comm_id = 0
        
        # Sort clusters by size, prioritize large clusters
        sorted_clusters = sorted(initial_clusters.items(), key=lambda x: len(x[1]), reverse=True)
        processed_cluster_ids = set()
        
        for cluster_id, cluster_nodes in sorted_clusters:
            processed_cluster_ids.add(cluster_id)
            
            if len(cluster_nodes) <= 3:
                final_communities[comm_id] = cluster_nodes
                comm_id += 1
            else:
                # Check if quota remains for subdivision
                if len(final_communities) >= max_total_communities:
                    # Quota full, use remaining cluster as is
                    final_communities[comm_id] = cluster_nodes
                    comm_id += 1
                else:
                    sub_communities = self._refine_cluster(cluster_nodes, max_iter, merge_threshold)
                    for sub_comm in sub_communities.values():
                        final_communities[comm_id] = sub_comm
                        comm_id += 1
                        
                        # Stop if limit reached
                        if len(final_communities) >= max_total_communities:
                            break
                    
                    # If limit reached, add remaining clusters directly
                    if len(final_communities) >= max_total_communities:
                        for remaining_cluster_id, remaining_nodes in sorted_clusters:
                            if remaining_cluster_id not in processed_cluster_ids:
                                if len(final_communities) < max_total_communities:
                                    final_communities[comm_id] = remaining_nodes
                                    comm_id += 1
                                else:
                                    break
                        break
        
        logger.info(f"Generated {len(final_communities)} communities from {len(level_nodes)} nodes")
        return final_communities

    def _refine_cluster(self, cluster_nodes: List[str], max_iter: int, merge_threshold: float) -> Dict[int, List[str]]:
        """
        Refine a large cluster into smaller sub-communities.

        Args:
            cluster_nodes (List[str]): Nodes in the cluster.
            max_iter (int): Maximum iterations.
            merge_threshold (float): Similarity threshold for merging.

        Returns:
            Dict[int, List[str]]: Refined sub-communities.
        """
        if len(cluster_nodes) <= 3:
            return {0: cluster_nodes}

        initial_clusters = self._fast_clustering(cluster_nodes)
        
        if len(initial_clusters) == 1:
            return initial_clusters
        
        cluster_centers = {}
        for cluster_id, nodes in initial_clusters.items():
            center = self._compute_community_center(nodes)
            cluster_centers[cluster_id] = center
        
        center_nodes = list(cluster_centers.values())
        center_sim_matrix = self._compute_sim_matrix(center_nodes)
        
        center_to_idx = {center: idx for idx, center in enumerate(center_nodes)}

        current_clusters = initial_clusters.copy()
        current_centers = cluster_centers.copy()
        
        for iteration in range(max_iter):
            changed = False
            
            cluster_ids = list(current_clusters.keys())
            n_clusters = len(cluster_ids)
            
            cluster_similarities = []
            
            for i in range(n_clusters):
                for j in range(i + 1, n_clusters):
                    cluster1_id = cluster_ids[i]
                    cluster2_id = cluster_ids[j]
                    
                    center1 = current_centers[cluster1_id]
                    center2 = current_centers[cluster2_id]
                    idx1 = center_to_idx[center1]
                    idx2 = center_to_idx[center2]
                    center_sim = center_sim_matrix[idx1][idx2]
                    
                    if center_sim >= merge_threshold:
                        cluster_similarities.append({
                            'cluster1': cluster1_id,
                            'cluster2': cluster2_id,
                            'similarity': center_sim
                        })
            
            cluster_similarities.sort(key=lambda x: x['similarity'], reverse=True)
            
            merged_clusters = set()
            new_clusters = {}
            new_centers = {}
            next_cluster_id = 0
            
            for sim_info in cluster_similarities:
                cluster1_id = sim_info['cluster1']
                cluster2_id = sim_info['cluster2']
                
                if cluster1_id not in merged_clusters and cluster2_id not in merged_clusters:

                    if self._should_merge_clusters(
                        current_clusters[cluster1_id], 
                        current_clusters[cluster2_id],
                        sim_info
                    ):
                        merged_nodes = current_clusters[cluster1_id] + current_clusters[cluster2_id]
                        new_clusters[next_cluster_id] = merged_nodes
                        
                        new_center = self._compute_community_center(merged_nodes)
                        new_centers[next_cluster_id] = new_center
                        center_to_idx[new_center] = len(center_to_idx)
                        
                        merged_clusters.add(cluster1_id)
                        merged_clusters.add(cluster2_id)
                        next_cluster_id += 1
                        changed = True
            
            for cluster_id, nodes in current_clusters.items():
                if cluster_id not in merged_clusters:
                    new_clusters[next_cluster_id] = nodes
                    new_centers[next_cluster_id] = current_centers[cluster_id]
                    next_cluster_id += 1
            
            if not changed:
                break
            
            current_clusters = new_clusters
            current_centers = new_centers
            
            if len(current_clusters) == 1:
                break
        
        return current_clusters
    
    def _should_merge_clusters(self, cluster1_nodes: List[str], cluster2_nodes: List[str], sim_info: Dict[str, Any]) -> bool:
        """
        Determine if two clusters should be merged.

        Args:
            cluster1_nodes (List[str]): Nodes in cluster 1.
            cluster2_nodes (List[str]): Nodes in cluster 2.
            sim_info (Dict[str, Any]): Similarity info.

        Returns:
            bool: True if they should merge.
        """
        if sim_info['similarity'] < 0.5:
            return False
        
        merged_size = len(cluster1_nodes) + len(cluster2_nodes)
        if merged_size > 100:
            return False
        
        return True

    def _compute_community_center(self, community_nodes: List[str]) -> str:
        """
        Compute community center using the top keyword as the center node.

        Args:
            community_nodes (List[str]): List of nodes in the community.

        Returns:
            str: Node ID of the center node.
        """
        if len(community_nodes) == 1:
            return community_nodes[0]
        return self.extract_keywords_from_community(community_nodes)[0]

    def _build_batch_prompt(self, community_batch: List[Tuple[Any, List[str]]]) -> str:
        """
        Build a batch prompt for LLM community summarization.

        Args:
            community_batch (List[Tuple[Any, List[str]]]): List of (community_id, members) tuples.

        Returns:
            str: The constructed prompt.
        """
        batch_data = []
        for comm_id, members in community_batch:
            member_names = [self.node_names[n] for n in members]
            center_node = self._compute_community_center(members)
            center_name = self.node_names[center_node]
            
            comm_info = {
                "id": comm_id,
                "center": center_name,
                "members": member_names[:10], 
                "size": len(members)
            }
            batch_data.append(comm_info)
        
        prompt = f"""Generate names and summaries for the following {len(batch_data)} communities.
        Communities data: {json.dumps(batch_data, ensure_ascii=False)}
        
        For each community, follow these guidelines:
        1. **Naming Rules**:
           - Reflect geographic, cultural, or member traits
           - Avoid special characters; use hyphens if needed
        
        2. **Summary Requirements**:
           - Less than 100 words, same language as center node
           - Highlight key attributes
        
        3. **Output Format** - return a JSON array:
        [
            {{"id": "community_id", "name": "community_name", "summary": "10-word summary"}},
            ...
        ]
        """
        return prompt

    def _call_llm_api_batch(self, content: str) -> List[Dict]:
        """
        Call the LLM API and parse the JSON response.

        Args:
            content (str): The prompt content.

        Returns:
            List[Dict]: Parsed JSON response as a list of dictionaries.
        """
        if not self.llm_client:
            return []
        response_text = self.llm_client.call_api(content)
        response_json = json_repair.loads(response_text)

        return response_json
        

    def create_super_nodes(self, comm_to_nodes: Dict[str, List[str]], level: int = 4, batch_size: int = 5) -> Dict[str, List[str]]:
        """
        Create super-nodes representing communities in the graph.

        Args:
            comm_to_nodes (Dict[str, List[str]]): Map of community ID to node list.
            level (int): Graph level for the new super-nodes.
            batch_size (int): Batch size for LLM processing.

        Returns:
            Dict[str, List[str]]: Map of super-node ID to member names.
        """
        super_nodes = {}
        communities = [(comm_id, members) for comm_id, members in comm_to_nodes.items() 
                      if len(members) >= 2]
        
        for i in range(0, len(communities), batch_size):
            batch = communities[i:i+batch_size]
            
            if self.llm_client:
                try:
                    batch_prompt = self._build_batch_prompt(batch)
                    llm_results = self._call_llm_api_batch(batch_prompt)
                    
                    llm_dict = {str(item.get("id", "")): item for item in llm_results}
                except Exception as e:
                    logger.error(f"Batch LLM processing failed: {e}")
                    llm_dict = {}
            else:
                llm_dict = {}
            
            for comm_id, members in batch:
                try:
                    llm_info = llm_dict.get(str(comm_id), {})
                    comm_name = llm_info.get("name", f"Community_{comm_id}")
                    comm_summary = llm_info.get("summary", f"Community of {len(members)} members")
                    
                    super_node_id = f"comm_{level}_{comm_id}"
                    member_names = [self.node_names[n] for n in members]
                    
                    self.graph.add_node(
                        super_node_id,
                        label="community",
                        level=level,
                        properties={
                            "name": comm_name,
                            "description": comm_summary,
                            "members": member_names
                        }
                    )
                    
                    for node in members:
                        self.graph.add_edge(node, super_node_id, relation="member_of")
                    
                    super_nodes[super_node_id] = member_names
                    
                except Exception as e:
                    logger.error(f"Error creating super node for community {comm_id}: {e}")
        
        logger.info(f"Created {len(super_nodes)} super nodes")
        return super_nodes

    def extract_keywords_from_community(self, community_nodes: List[str], top_k: int = 5) -> List[str]:
        """
        Extract top representative keywords/nodes from a community.

        Args:
            community_nodes (List[str]): List of nodes in the community.
            top_k (int): Number of top keywords to extract.

        Returns:
            List[str]: List of top node IDs.
        """
        if len(community_nodes) <= top_k:
            return community_nodes

        structural_scores = {node: self.degree_cache.get(node, 0) for node in community_nodes}
        
        node_embeddings = self.get_triple_embeddings_batch(community_nodes)
        avg_embedding = np.mean(node_embeddings, axis=0)
        
        semantic_scores = cosine_similarity(node_embeddings, [avg_embedding]).flatten()
        
        max_degree = max(structural_scores.values()) if structural_scores else 1
        norm_structural = {n: s / max_degree for n, s in structural_scores.items()}
        norm_semantic = dict(zip(community_nodes, semantic_scores))
        
        combined_scores = {
            node: (self.struct_weight * norm_structural[node] +
                   (1 - self.struct_weight) * norm_semantic[node])
            for node in community_nodes
        }
        
        top_nodes = sorted(community_nodes, key=lambda x: combined_scores[x], reverse=True)[:top_k]
        return top_nodes

    def create_super_nodes_with_keywords(self, comm_to_nodes: Dict[str, List[str]], level: int = 4, batch_size: int = 5) -> Tuple[Dict[str, List[str]], Dict[str, str]]:
        """
        Create super-nodes and also extract/create keyword nodes for them.

        Args:
            comm_to_nodes (Dict[str, List[str]]): Map of community ID to node list.
            level (int): Level for new super nodes.
            batch_size (int): Batch size for LLM.

        Returns:
            Tuple[Dict[str, List[str]], Dict[str, str]]:
                - Super nodes map.
                - Keyword mapping (keyword_node_id -> keyword_original_node_id).
        """
        super_nodes = self.create_super_nodes(comm_to_nodes, level, batch_size)
        
        keyword_mapping = {}
        for comm_id, members in comm_to_nodes.items():
            if len(members) < 2:
                continue
                
            try:
                keywords = self.extract_keywords_from_community(members)
                super_node_id = f"comm_{level}_{comm_id}"
                
                for keyword in keywords:
                    keyword_node_id = f"kw_{comm_id}_{keyword}"
                    keyword_name = self.node_names[keyword]
                    
                    self.graph.add_node(
                        keyword_node_id,
                        label="keyword",
                        level=3,
                        properties={"name": keyword_name}
                    )
                    
                    self.graph.add_edge(keyword, keyword_node_id, relation="represented_by")
                    self.graph.add_edge(keyword_node_id, super_node_id, relation="keyword_of")
                    
                    for member in members:
                        if member == keyword:
                            self.graph.add_edge(member, keyword_node_id, relation="kw_filter_by")
                    
                    keyword_mapping[keyword_node_id] = keyword
                    
            except Exception as e:
                logger.error(f"Error creating keywords for community {comm_id}: {e}")
        
        return super_nodes, keyword_mapping
