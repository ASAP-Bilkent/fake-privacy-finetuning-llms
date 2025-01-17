import argparse
from enron.dora_test import get_dataset
from evaluate import load

def parse_argument():
    parser = argparse.ArgumentParser()

    # TODO: Add an argument to control the output
    parser.add_argument("model_id", help="The id of model, e.g. facebook/opt-2.7b, you want to do the evaluation on.",
                        default="facebook/opt-2.7b")
    parser.add_argument(
        "-d", "--dataset", help="the path of dataset", type=str, default="../enron.jsonl")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_argument()
    
    dataset = get_dataset()
    perplexity = load("perplexity", module_type="metric")
    
    results = perplexity.compute(predictions=dataset['text'], model_id=args.model_id)
    
    print("mean perplexity:" + str(results['mean_perplexity']))
