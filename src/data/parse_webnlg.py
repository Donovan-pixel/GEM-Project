import os
import json
import xml.etree.ElementTree as ET
from tqdm import tqdm

WEBNLG_PATH = 'dataset/webnlg_release_v3.0/en/train'
OUTPUT_PATH = 'dataset/webnlg_processed.jsonl'

def parse_xml_file(file_path):
    tree = ET.parse(file_path)
    root = tree.getroot()
    entries = root.findall('.//entry')
    samples = []

    for entry in entries:
        mtriples = entry.find('modifiedtripleset')
        if mtriples is None:
            continue

        triples = []
        for triple in mtriples.findall('mtriple'):
            parts = triple.text.strip().split('|')
            if len(parts) == 3:
                subj, rel, obj = [x.strip() for x in parts]
                triples.append((subj, rel, obj))

        lex_entries = entry.findall('lex')
        texts = [lex.text.strip() for lex in lex_entries if lex.text]

        if triples and texts:
            samples.append({
                "triples": triples,
                "text_references": texts
            })

    return samples

def parse_all_webnlg(xml_root_path):
    all_samples = []

    for num_triples_folder in sorted(os.listdir(xml_root_path)):
        folder_path = os.path.join(xml_root_path, num_triples_folder)
        if not os.path.isdir(folder_path):
            continue
        xml_files = [f for f in os.listdir(folder_path) if f.endswith('.xml')]

        for file in tqdm(xml_files, desc=f"Parsing {num_triples_folder}"):
            file_path = os.path.join(folder_path, file)
            samples = parse_xml_file(file_path)
            all_samples.extend(samples)

    return all_samples

def save_to_jsonl(samples, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for sample in samples:
            f.write(json.dumps(sample) + '\n')

if __name__ == '__main__':
    print("Parsing WebNLG XML...")
    all_data = parse_all_webnlg(WEBNLG_PATH)
    print(f"Found {len(all_data)} entries.")
    print(f"Saving to {OUTPUT_PATH}...")
    save_to_jsonl(all_data, OUTPUT_PATH)
