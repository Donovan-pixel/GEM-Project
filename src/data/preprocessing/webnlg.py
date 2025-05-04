import os
import json
import torch
from torch_geometric.data import Dataset, Data
import random


class WebNLGGraphTextDataset(Dataset):
    """
    Dataset combinant les graphes RDF (PyG) et les textes cibles de WebNLG pour l'entraînement ou l'inférence.
    Chaque échantillon retourne un objet `Data` enrichi de :
        - .triples : la liste des triplets RDF (liste de tuples)
        - .desc : les triplets sérialisés en chaîne (entrée LLM)
        - .label : la phrase cible (ou liste de phrases en validation/test)
        - .id : identifiant de l'échantillon
    """

    def __init__(self, graph_dir: str, jsonl_path: str, split: str = "train", use_random_ref: bool = True):
        super().__init__()
        self.split = split
        self.graph_dir = graph_dir
        self.use_random_ref = use_random_ref

        # Liste des fichiers de graphes PyG
        self.graph_files = sorted([f for f in os.listdir(graph_dir) if f.endswith('.pt')])

        # Chargement du JSONL contenant les triples et les phrases
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            self.entries = [json.loads(line) for line in f]

        assert len(self.graph_files) == len(self.entries), (
            f"Mismatch between {split} graph files ({len(self.graph_files)}) and JSONL entries ({len(self.entries)})"
        )

    def len(self):
        return len(self.graph_files)

    def get(self, idx):
        # Chargement du graphe PyG
        graph_path = os.path.join(self.graph_dir, self.graph_files[idx])
        graph_data = torch.load(graph_path, map_location='cpu', weights_only=False)

        # Chargement des informations textuelles
        entry = self.entries[idx]
        graph_data.triples = entry["triples"]
        graph_data.id = idx

        # Construction de l'entrée textuelle pour le LLM
        graph_data.desc = self.serialize_triples(entry["triples"])

        # Sélection d'une phrase cible pour supervision
        text_refs = entry.get("text_references", [])
        if text_refs:
            if self.split == "train" and self.use_random_ref:
                graph_data.label = random.choice(text_refs)
            else:
                graph_data.label = text_refs  # liste complète pour évaluation
        else:
            graph_data.label = None  # Pas de référence dans le test set sans refs

        return graph_data

    @staticmethod
    def serialize_triples(triples):
        """
        Sérialise une liste de triples RDF en une chaîne lisible pour le LLM.
        """
        return " ; ".join([f"{s} | {p} | {o}" for s, p, o in triples])
