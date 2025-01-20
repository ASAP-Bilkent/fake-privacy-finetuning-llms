import argparse
import os
from types import SimpleNamespace

from merge_peft import run_merge_peft
from measure_purified import run_measure_purified


def parse_argument():
    parser = argparse.ArgumentParser()

    parser.add_argument("-m", "--model", help="Model to be ran.",
                        default="./pythia-2.8b")
    parser.add_argument("-t", "--temp", help="temporary storage in which the adapter and the base model will be merged.",
                        default="./tempMerged")
    parser.add_argument(
        "-d", "--dataset", help="the path of dataset", type=str, default="./enron.jsonl")
    parser.add_argument("--is_opt", action='store_true', help="Determines if the model is an opt variant.",
                        default=False)

    return parser.parse_args()


def process_checkpoint(checkpoint_path, temporaryStorage, dataset):
    run_merge_peft(SimpleNamespace(**{
        "model": checkpoint_path,
        "save_dir": temporaryStorage,
        "is_opt": args.is_opt
    }))

    run_measure_purified(SimpleNamespace(**{
        "model": temporaryStorage,
        "output_dir": checkpoint_path,
        "dataset": dataset,
        "top_k": 100,
        "max_tokens": 15
    }))

    print(checkpoint_path + "has been finished")


def traverse_checkpoints(args):
    base_path = args.model

    for root, dirs, _ in os.walk(base_path):
        for dir_name in dirs:
            if dir_name.startswith("checkpoint-"):
                checkpoint_path = os.path.join(root, dir_name)
                if (not os.path.isfile(os.path.join(checkpoint_path, 'results_purified.jsonl'))):
                    print(checkpoint_path + ' is under process.')
                    process_checkpoint(
                        checkpoint_path, args.temp, args.dataset)


if __name__ == "__main__":
    args = parse_argument()
    traverse_checkpoints(args)
