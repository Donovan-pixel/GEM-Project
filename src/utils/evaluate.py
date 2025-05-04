import json
import nltk
import re
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score

nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)


def normalize(text):
    text = text.lower()
    text = re.sub(r"<pad>", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def get_accuracy_webnlg(path):
    bleu_scores = []
    meteor_scores = []

    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            entry = json.loads(line)
            pred = normalize(entry["pred"])
            refs = entry["label"]

            # Ensure refs is a list of strings
            if isinstance(refs, str):
                refs = [refs]
            elif isinstance(refs, list):
                refs = [normalize(ref) for ref in refs]
            else:
                continue

            # BLEU (up to 4-gram with smoothing)
            bleu = sentence_bleu(
                [ref.split() for ref in refs],
                pred.split(),
                weights=(0.25, 0.25, 0.25, 0.25),
                smoothing_function=SmoothingFunction().method4
            )
            bleu_scores.append(bleu)

            # METEOR
            try:
                meteor = max(meteor_score([ref], pred) for ref in refs)
            except:
                meteor = 0.0
            meteor_scores.append(meteor)

    avg_bleu = 100 * sum(bleu_scores) / len(bleu_scores)
    avg_meteor = 100 * sum(meteor_scores) / len(meteor_scores)

    print(f"BLEU-4: {avg_bleu:.2f}")
    print(f"METEOR: {avg_meteor:.2f}")

    return {
        "BLEU": avg_bleu,
        "METEOR": avg_meteor,
    }

eval_funcs = {
    'webnlg': get_accuracy_webnlg,
}