import torch
from torch_geometric.data import Data

# Here you can select the desired file between 0 and 13210
graph = torch.load("dataset/webnlg/graphs/1.pt", weights_only=False)

# Print a summary of the graph object: number of nodes, edges, and the shape of each component
print(graph)

# Print the node features (also called node embeddings)
# graph.x is a tensor of shape [num_nodes, embedding_dim]
# Each row is a vector representing a node in the graph (e.g., an entity like "Aarhus_Airport")
print("\nNode features (x):")
print(graph.x)  # Tensor [num_nodes, embedding_dim]

# Print the edge index tensor, which defines the graph structure (i.e., connections)
# edge_index is a tensor of shape [2, num_edges]
# Each column defines a directed edge: from node edge_index[0, i] to node edge_index[1, i]
print("\nEdge index:")
print(graph.edge_index)  # Tensor [2, num_edges]

# Print the edge attributes (relation embeddings between nodes)
# edge_attr is a tensor of shape [num_edges, embedding_dim]
# Each row represents the textual embedding of the RDF relation (e.g., "location")
print("\nEdge attributes (relations):")
print(graph.edge_attr)  # Tensor [num_edges, embedding_dim]

# Print the total number of nodes in the graph
print(f"\nNumber of nodes: {graph.num_nodes}")
