from transformers import AutoTokenizer
from datasets import load_dataset
import torch
import os
from vllm import LLM, SamplingParams
import argparse
import re
import subprocess

# TODO: this script needs refactoring

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")

    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    dataset = load_dataset("json", data_files=args.data_set, split="train")

    dataset = dataset.select(range(1110, len(dataset)))

    # print(dataset["prompt"][573])

    def createPrompts(x):
        x['prompt'] = x['prompt'] + " "

        return x

    dataset = dataset.map(createPrompts)

    dataset = dataset.filter(lambda x: len(
        tokenizer.encode(x["prompt"])) < 2048)

    model_name = args.model

    model = LLM(
        model_name,
        trust_remote_code=True,
        tensor_parallel_size=2,
    )

    sampling_params = SamplingParams(
        temperature=0.75,
        top_k=args.top_k,
        max_tokens=args.max_tokens,
    )

    generations = model.generate(dataset['prompt'], sampling_params)

    def saveAnswers(example, index):
        example["generation"] = generations[index].outputs[0].text

        return example

    dataset = dataset.map(saveAnswers, with_indices=True)

    def concatenate(x):
        x['text'] = x['prompt'] + x['generation'] + tokenizer.eos_token

        return x

    dataset = dataset.map(concatenate)

    # name = re.search(r'/(pythia[^/]*)', args.model).group(1)
    name = re.search(r'/(pythia[^/]*|opt-[^/]*)',
                     args.model).group(1) + "-new-90"

    dataset.save_to_disk(name)

    subprocess.run(f"ls -l ./{name} | grep -E \.arrow | tr -s ' ' | cut -d ' ' -f 9 | xargs -I _ mv ./{
                   name}/_ ./{name}/data.arrow", shell=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", help="name of the model to be evaluated",
                        default="EleutherAI/pythia-12b")
    parser.add_argument("--top_k", help="add vllm doc",
                        type=int, default=100)
    parser.add_argument("-mt", "--max_tokens", help="add vllm doc",
                        type=int, default=1000)
    parser.add_argument("-d", "--data_set", help="dataset path",
                        type=str, default="../enron.jsonl")
    args = parser.parse_args()

    main(args)
