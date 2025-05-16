#!python3
import argparse
import json
from evaluate import load

parser = argparse.ArgumentParser()
parser.add_argument("-o", "--out_file", type=str)
#arser.add_argument("-g", "--gold_file", type=str)
parser.add_argument("-gpu", action="store_true")

def load_data(out_file):
    gold, pred = [], []
    with open(out_file, "r") as r:
        for line in r:
            data = json.loads(line.strip())
            gold.append(data["label"])
            pred.append(data["pred"])
    return gold, pred

def score(gold, test, metrics):
    if metrics in ["bleu"]:
        score = load(metrics)
        results = score.compute(predictions=test, references=gold)
    elif metrics in ["exact_match"]:
        ngold = [a if any(z == a for z in ref) else ref[0] for a, ref in zip(test, gold)]
        score = load(metrics)
        results = score.compute(predictions=test, references=ngold)
    elif metrics in ["bertscore"]:
        score = load(metrics)
        results = score.compute(predictions=test, references=gold, lang="en")
        results["avg_precision"] = sum(results["precision"]) / len(results["precision"])
    return (metrics, results)

def score_all(gold, test, gpu=False):
    metrics_list = ["bleu", "exact_match"]
    if gpu:
        metrics_list += ["bertscore"]
    results = [score(gold, test, metrics) for metrics in metrics_list]
    return results

if __name__ == "__main__":
    args = parser.parse_args()
    gold, test = load_data(args.out_file)
    print("Evaluation...")
    results = score_all(gold, test, args.gpu)
    print("Evaluation results")
    for item in results:
        print(item[0] + "\t" + str(item[1]))


