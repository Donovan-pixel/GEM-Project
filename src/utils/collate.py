from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader
from src.data.preprocessing.webnlg import WebNLGGraphTextDataset

def graph_llm_collate_fn(batch):
    """
    Combine une liste d'échantillons WebNLG en un batch LLM + GNN.
    Chaque item doit être un objet Data avec les attributs : 
    - desc (str) : description RDF sérialisée pour le LLM
    - label (str or List[str]) : phrases cibles
    - id : identifiant
    - question (str) : prompt de génération
    """
    # Fusionner les graphes en un batch unique (PyG)
    batched_graph = Batch.from_data_list(batch)

    # Extraire les autres infos sous forme de liste
    batched_graph.desc = [data.desc for data in batch]
    batched_graph.label = [data.label for data in batch]
    batched_graph.id = [data.id for data in batch]
    batched_graph.question = ["Generate a natural language sentence that describes the following RDF graph:" for _ in batch]

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
