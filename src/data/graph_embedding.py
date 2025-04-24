import os
import torch
import umap.umap_ as umap
import plotly.express as px
from tqdm import tqdm

graph_dir = "dataset/webnlg/graphs"
graph_files = sorted([f for f in os.listdir(graph_dir) if f.endswith('.pt')])
embeddings = []
file_ids = []

for f in tqdm(graph_files[:13210]):
    graph = torch.load(os.path.join(graph_dir, f), weights_only=False)
    graph_embedding = graph.x.mean(dim=0)
    embeddings.append(graph_embedding.numpy())
    file_ids.append(f.split(".")[0])

# UMAP en 3D
reducer = umap.UMAP(n_components=3, n_neighbors=30, min_dist=0.05, metric='cosine', n_jobs=-1)
embedding_3d = reducer.fit_transform(embeddings)

fig = px.scatter_3d(
    x=embedding_3d[:, 0],
    y=embedding_3d[:, 1],
    z=embedding_3d[:, 2],
    hover_name=file_ids,
    title="UMAP 3D of WebNLG Graph Embeddings",
    opacity=0.7
)
fig.show()
