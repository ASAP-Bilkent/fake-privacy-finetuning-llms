from evaluate import load
from datasets import load_dataset, Dataset
import torch
import argparse
import os
import sys
import logging

# TODO: refactor this.

_original_stdout = sys.stdout
_original_stderr = sys.stderr


def block_output():
    # Ensure output is directed to the console
    sys.stdout = _original_stdout
    sys.stderr = _original_stderr
    logging.disable(logging.NOTSET)


def enable_output(dir):
    sys.stdout = open(dir, 'a')
    sys.stderr = open(dir, 'a')
    logging.disable(logging.NOTSET)


def process_checkpoint(checkpoint_path, output_dir):
    block_output()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = load_dataset("json", data_files='../enron.jsonl', split="train")

    dataset = dataset.select(range(0, 1110))

    def createText(x):
        x['text'] = x['prompt'] + x['continuation']

        return x

    dataset = dataset.map(createText)

    perplexity = load("perplexity", module_type="metric")
    results = perplexity.compute(
        predictions=dataset['text'], model_id=checkpoint_path)

    enable_output(output_dir)

    print("model path: " + str(checkpoint_path) +
          "; mean perplexity:" + str(results['mean_perplexity']))

    block_output()


def traverse_checkpoints(base_path, output_dir):
    for root, dirs, files in os.walk(base_path):
        if "/self-generated-dora-pythia-12b/" in root:
            for dir_name in dirs:
                full_path = os.path.join(root, dir_name)
                if dir_name.startswith("checkpoint-"):
                    print(f"root is {full_path}")
                    process_checkpoint(full_path, output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("dir", help="Directory that includes the models to be ran.",
                        default="../pythia-1.4b")

    args = parser.parse_args()

    os.makedirs(args.dir, exist_ok=True)

    traverse_checkpoints(args.dir, f"{args.dir}/perplexity.txt")
