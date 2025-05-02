#!python3
import argparse
import json
from evaluate import load
import log

parser = argparse.ArgumentParser()
parser.add("-s", "--source_file", type=str)
parser.add("-t", "--target_file", type=str)
parser.add("-gpu", action="store_true")

def load_data(gold_file, test_file):
    gold, test = [], []
    with open(gold_file, "r") as r:
        for line in r:
            gold.append(line.strip())
    with open(test_file, "r") as r:
        for line in r:
            data = json.loads(line.strip())
            test.append(data["test_references"])
    return gold, test
          
def score(gold, test, metrics):
    bertscore = load(metrics)
    results = bertscore.compute(predictions=test, references=gold, lang="en")
    return(metrics, results)

def score_all(gold, test, gpu=False):
    metrics_list = ["bleu", "chrf++", "exact_match"]
    if gpu:
        metrics_list += ["bertscore"]
    results = [score(gold, test, metrics) for metrics in metrics_list]
    return results

if __name__ == "__main__":
    parser = argparse.parse_args()
    gold, test = load_data(args.s, args.t)
    print("Evaluation...")
    results = score_all(gold, test, args.gpu)
    print("Evaluation results")
    for item in results:
        print(item[0] + "\t" + str(item[1]))


