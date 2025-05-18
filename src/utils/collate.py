from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader
from src.data.preprocessing.webnlg import WebNLGGraphTextDataset

def graph_llm_collate_fn(batch):
    """
    Combine a list of WebNLG samples into a batch for LLM + GNN.
    Each item should be a Data object with attributes:
    - desc (str): serialized RDF description for the LLM
    - label (str or List[str]): target sentences
    - id: identifier
    - question (str): generation prompt
    """
    # Merge graphs into a single batch (PyG)
    batched_graph = Batch.from_data_list(batch)

    # Extract other info as lists
    batched_graph.desc = [data.desc for data in batch]
    batched_graph.label = [data.label for data in batch]
    batched_graph.id = [data.id for data in batch]
    batched_graph.question = ["Generate a natural language sentence that describes the following RDF graph:" for _ in batch]

    # Add the graph info into the batch as well (for model.forward compatibility)
    batch_dict = {
        'desc': batched_graph.desc,
        'label': batched_graph.label,
        'id': batched_graph.id,
        'question': batched_graph.question,
        'x': batched_graph.x,
        'edge_index': batched_graph.edge_index,
        'edge_attr': batched_graph.edge_attr,
        'batch': batched_graph.batch
    }

    # Return both the PyG Batch object and the dict
    # The model will be updated to handle either format
    return batched_graph

train_dataset = WebNLGGraphTextDataset(
    graph_dir="dataset/webnlg/train",
    jsonl_path="dataset/train.jsonl",
    split="train"
)

train_loader = DataLoader(
    train_dataset,
    batch_size=8,
    shuffle=True,
    collate_fn=graph_llm_collate_fn
)

# for batch in train_loader:
#     print("RDF Descriptions for LLM:", batch.desc)
#     print("Text targets:", batch.label)
#     print("Graph node features shape:", batch.x.shape)
#     print("Edge index:", batch.edge_index.shape)
#     print("Batch vector:", batch.batch.shape)
#     print(batch.edge_attr)  # indique à quel graphe chaque nœud appartient
#     break
