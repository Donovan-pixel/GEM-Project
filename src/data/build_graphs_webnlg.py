import os
import json
import torch
import pandas as pd
from tqdm import tqdm
from torch_geometric.data import Data

from src.data.utils.lm_modeling import load_model, load_text2embedding

INPUT_PATH = 'dataset/webnlg_processed.jsonl'
GRAPH_DIR = 'dataset/webnlg/graphs'
model_name = 'sbert'  

model, tokenizer, device = load_model[model_name]()
text2embedding = load_text2embedding[model_name]

def process_sample(entry):
    triples = entry['triples']

    nodes = {}
    edges = []

    for subj, rel, obj in triples:
        subj = subj.lower().strip()
        obj = obj.lower().strip()
        rel = rel.lower().strip()

        if subj not in nodes:
            nodes[subj] = len(nodes)
        if obj not in nodes:
            nodes[obj] = len(nodes)

        edges.append({'src': nodes[subj], 'edge_attr': rel, 'dst': nodes[obj]})

    node_list = list(nodes.keys())
    rel_list = [e['edge_attr'] for e in edges]

    x = text2embedding(model, tokenizer, device, node_list)
    edge_attr = text2embedding(model, tokenizer, device, rel_list)
    edge_index = torch.tensor([[e['src'] for e in edges], [e['dst'] for e in edges]], dtype=torch.long)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, num_nodes=len(nodes))

def process_all():
    os.makedirs(GRAPH_DIR, exist_ok=True)

    with open(INPUT_PATH, 'r', encoding='utf-8') as f:
        entries = [json.loads(line) for line in f.readlines()]

    for i, entry in tqdm(enumerate(entries), total=len(entries), desc="Building PyG graphs"):
        try:
            data = process_sample(entry)
            torch.save(data, f'{GRAPH_DIR}/{i}.pt')
        except Exception as e:
            print(f"[!] Skipping index {i} due to error: {e}")

if __name__ == '__main__':
    process_all()
