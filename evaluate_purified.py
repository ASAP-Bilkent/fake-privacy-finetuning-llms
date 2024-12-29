import argparse
import os

from merge_peft import run_merge_peft
from measure_purified import run_measure_purified


def process_checkpoint(checkpoint_path):
    temporaryStorage = args.temp

    run_merge_peft({
        "model": checkpoint_path,
        "save_dir": temporaryStorage,
        "is_opt": True
    })

    run_measure_purified({
        "model": temporaryStorage,
        "output_dir": checkpoint_path
    })

    print(checkpoint_path + "has been finished")


def traverse_checkpoints(base_path):
    for root, dirs, _ in os.walk(base_path):
        for dir_name in dirs:
            if dir_name.startswith("checkpoint-"):
                checkpoint_path = os.path.join(root, dir_name)
                if (not os.path.isfile(os.path.join(checkpoint_path, 'results_purified.jsonl'))):
                    print(checkpoint_path + ' is under process.')
                    process_checkpoint(checkpoint_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-m", "--model", help="Model to be ran.",
                        default="./pythia-2.8b")
    parser.add_argument("-t", "--temp", help="temporary storage in which the adapter and the base model will be merged.",
                        default="./tempMerged")

    args = parser.parse_args()
    traverse_checkpoints(args.model)
