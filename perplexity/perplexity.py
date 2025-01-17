from evaluate import load
from datasets import load_dataset
import argparse
import os
import sys
import logging


_original_stdout = sys.stdout
_original_stderr = sys.stderr


def parse_argument():
    parser = argparse.ArgumentParser()

    # TODO: Add an argument to control the output
    parser.add_argument("dir", help="Directory that includes the models to be ran.",
                        default="../pythia-1.4b")

    return parser.parse_args()


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
    
    dataset = load_dataset("json", data_files='../enron.jsonl', split="train")
    dataset = dataset.select(range(0, 1110))
    dataset = dataset.map(lambda x: x['prompt'] + x['continuation'])

    perplexity = load("perplexity", module_type="metric")
    results = perplexity.compute(predictions=dataset['text'], model_id=checkpoint_path)

    enable_output(output_dir)

    # TODO: replace this with a rational system.
    print("model path: " + str(checkpoint_path) +"; mean perplexity:" + str(results['mean_perplexity']))

    block_output()


def traverse_checkpoints(base_path, output_dir):
    for root, dirs, _ in os.walk(base_path):
        for dir_name in dirs:
            if dir_name.startswith("checkpoint-"):
                checkpoint_path = os.path.join(root, dir_name)
                print(f"root is {checkpoint_path}")
                process_checkpoint(checkpoint_path, output_dir)


if __name__ == "__main__":
    args = parse_argument()

    os.makedirs(args.dir, exist_ok=True)

    traverse_checkpoints(args.dir, f"{args.dir}/perplexity.txt")
