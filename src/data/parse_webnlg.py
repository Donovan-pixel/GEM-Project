import os
import json
import xml.etree.ElementTree as ET
from tqdm import tqdm

INPUT_XML = 'dataset/webnlg_release_v3.0/en/test/rdf-to-text-generation-test-data-with-refs-en.xml'
OUTPUT_JSONL = 'dataset/test.jsonl'


def parse_webnlg_test_file(xml_file):
    tree = ET.parse(xml_file)
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

        texts = [lex.text.strip() for lex in entry.findall('lex') if lex.text]

        if triples and texts:
            samples.append({
                "triples": triples,
                "text_references": texts
            })

    return samples


def save_to_jsonl(samples, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for sample in samples:
            f.write(json.dumps(sample) + '\n')


if __name__ == '__main__':
    print("Parsing RDF-to-Text WebNLG Test File with Refs...")
    parsed_samples = parse_webnlg_test_file(INPUT_XML)
    print(f"✅ Extracted {len(parsed_samples)} test entries")
    save_to_jsonl(parsed_samples, OUTPUT_JSONL)
    print(f"✅ Saved to {OUTPUT_JSONL}")
