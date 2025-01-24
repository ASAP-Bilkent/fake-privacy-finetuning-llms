from transformers import AutoModelForCausalLM, AutoTokenizer
import re
from argparse import ArgumentParser
from peft import PeftModel

# TODO: add wandb's logging.


def parse_args():
    parser = ArgumentParser()

    parser.add_argument("--model")
    parser.add_argument("--save_dir", default="./tempMerged")
    parser.add_argument("--is_opt", action='store_true', help="Determines if the model is an opt variant.",
                        default=False)

    return parser.parse_args()


def run_merge_peft(args):
    model_path = args.model
    print(f"Merging PEFT model started. Model path: {model_path}")

    print("Loading model.")
    if args.is_opt:
        model_name = "facebook/" + \
            re.search(r'/(opt-[^/]*)/', model_path).group(1)
    else:
        model_name = "EleutherAI/" + \
            re.search(r'/(pythia[^/]*)/', model_path).group(1)

    base_model = AutoModelForCausalLM.from_pretrained(model_name)

    tokenizer = AutoTokenizer.from_pretrained(model_path)

    tokenizer.save_pretrained(args.save_dir)
    peft_model = PeftModel.from_pretrained(base_model, model_path)
    merged_model = peft_model.merge_and_unload()

    merged_model.save_pretrained(args.save_dir)
    tokenizer.save_pretrained(args.save_dir)
    
    print("Merging finished successfully.")


if __name__ == "__main__":
    args = parse_args()

    # TODO: fix this.
    if args.is_opt:
        print("running tasks for OPT")
    else:
        print("running tasks for PyThia")

    run_merge_peft(args)
