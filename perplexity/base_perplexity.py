from evaluate import load
from datasets import load_dataset
import torch

# TODO: refactor this.

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = load_dataset("json", data_files='../enron.jsonl', split="train")

dataset = dataset.select(range(0, 1110))

def createText(x):
    x['text'] = x['prompt'] + x['continuation']

    return x

dataset = dataset.map(createText)

perplexity = load("perplexity", module_type="metric")
results = perplexity.compute(predictions=dataset['text'], model_id='facebook/opt-2.7b')

print("mean perplexity:" + str(results['mean_perplexity']))
