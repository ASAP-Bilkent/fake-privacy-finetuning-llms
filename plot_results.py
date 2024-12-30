from datasets import Dataset
import re
import os
import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd


HOME_DIR = "/home/c01mili/CISPA-projects/llm_ftsec-2024/atilla/enron"
OTPIMAL_TEMPRATURE = 0.2
OPTIMAL_LR = 2e-6
OPTIMAL_EPOCH = "checkpoint-276"
INCLUCE_ORIGINAL = False
FINETUNE_TYPE = "self-generated-pissa-pythia-12b"

BASE_PATHES = [
    '/home/c01mili/CISPA-projects/llm_ftsec-2024/atilla/enron/opt-1.3b',
    '/home/c01mili/CISPA-projects/llm_ftsec-2024/atilla/enron/opt-2.7b',
]

BASE_RESULTS = [
    {
        'name': 'opt-1.3b',
        'A': 56,
        'B': 58,
        'C': 33,
        'D': 31
    },
    {
        'name': 'opt-2.7b',
        'A': 28,
        'B': 40,
        'C': 35,
        'D': 33
    }
]

MODELS = [
    'opt-1.3b',
    'opt-2.7b',
    # 'pythia-6.9b'
]


def perplexity_lr(data, name):
    learning_rates = [float(row['learning_rate']) for row in data]
    perplexities = [float(row['perplexity']) for row in data]

    linestyle = 'solid'
    color = 'blue'

    if ('deduped' in name):
        linestyle = 'dashed'
    if ('original' in name):
        color = 'red'

    plt.plot(learning_rates, perplexities, marker='o',
             label=name,  linestyle=linestyle, color=color)


def perplexity_epoch(data, name):
    a = {
        'checkpoint-69': 1,
        'checkpoint-138': 2,
        'checkpoint-208': 3,
        'checkpoint-276': 4,
    }

    data = data.map(lambda x: {'checkpoint': a[x['checkpoint']]})
    data = data.sort('checkpoint')

    epochs = data['checkpoint']
    perplexities = [float(row['perplexity']) for row in data]

    linestyle = 'solid'
    color = 'blue'

    if ('deduped' in name):
        linestyle = 'dashed'
    if ('original' in name):
        color = 'red'

    plt.plot(epochs, perplexities, marker='o',
             label=name, linestyle=linestyle, color=color)


def parse_results(path: str):
    lines = []

    with open(path, 'r') as fs:
        lines = fs.read().split('\n')
        print(lines)

    # Extract model info
    path = path.replace(HOME_DIR, '.')  # to fix atilla's logic
    path = re.search('[^\.]*\./(.*)', path).group(1)
    path = path.split('/')

    base_model = path[0]
    finetune_type = path[1]
    weight_decay = float(re.search("(.*)_.*", path[2]).group(1))
    learning_rate = float(re.search(".*_(.*)", path[2]).group(1))
    checkpoint = path[3]

    result_lines = []

    for line in lines:
        if not line:
            continue

        processed_line = json.loads(line)
        print(processed_line)

        record = {}
        record["base_model"] = base_model
        record["finetune_type"] = finetune_type
        record["weight_decay"] = weight_decay
        record["learning_rate"] = learning_rate
        record["checkpoint"] = checkpoint
        record["temperature"] = processed_line["temperature"]
        record["top-k"] = processed_line["top_k"]

        for key in processed_line["matches"]:
            if key != "total":
                tmp_record = record.copy()
                tmp_record["template_type"] = key
                tmp_record["success"] = processed_line["matches"][key]
                result_lines.append(tmp_record)

    print(result_lines)
    return result_lines


def temperature_graph(dataset, model_name, ax):
    dataset = dataset.filter(lambda x: x['base_model'] == model_name)
    temperatures = dataset.unique('temperature')
    temperatures.sort()

    for (template, color) in zip(['A', 'B', 'C', 'D'], ['red', 'green', 'orange', 'blue']):
        ax.plot(
            temperatures,
            [np.max(dataset.filter(lambda x: (x['temperature'] == temp) and (x['template_type'] == template) and (
                x['finetune_type'] == FINETUNE_TYPE))['success']) for temp in temperatures],
            marker='o',
            label='Template ' + template,
            color=color
        )

    ax.set_xscale('log')
    ax.set_xlabel('Temperature', fontsize=14)
    ax.set_ylabel(model_name, fontsize=25)


def success_lr(dataset, model_name, ax):
    dataset = dataset.sort("learning_rate")
    dataset = dataset.filter(lambda x: x['base_model'] == model_name)
    learning_rates = dataset.unique('learning_rate')
    learning_rates.sort()

    for (template, color) in zip(['A', 'B', 'C', 'D'], ['red', 'green', 'orange', 'blue']):
        ax.plot(
            learning_rates,
            [np.max(dataset.filter(lambda x: (x['temperature'] == OTPIMAL_TEMPRATURE) and (x['template_type'] == template) and (
                x['learning_rate'] == lr) and (x['finetune_type'] == FINETUNE_TYPE))['success']) for lr in learning_rates],
            marker='o',
            label=template,
            color=color
        )

    ax.set_xscale('log')
    ax.set_xlabel('learning rate', fontsize=14)
    ax.set_ylabel(model_name, fontsize=25)


def success_epoch(dataset, model_name, ax):
    dataset = dataset.filter(lambda x: (
        x['base_model'] == model_name) and x['learning_rate'] == OPTIMAL_LR)

    a = {
        'checkpoint-69': 1,
        'checkpoint-138': 2,
        'checkpoint-208': 3,
        'checkpoint-276': 4,
    }

    dataset = dataset.map(lambda x: {'checkpoint': a[x['checkpoint']]})

    dataset = dataset.sort('checkpoint')

    epochs = dataset.unique('checkpoint')
    epochs.sort()

    for (template, color) in zip(['A', 'B', 'C', 'D'], ['red', 'green', 'orange', 'blue']):
        ax.plot(
            epochs,
            [np.max(dataset.filter(lambda x: (x['temperature'] == OTPIMAL_TEMPRATURE) and (x['template_type'] == template) and (x['learning_rate']
                    == OPTIMAL_LR) and (x['checkpoint'] == epoch) and (x['finetune_type'] == FINETUNE_TYPE))['success']) for epoch in epochs],
            marker='o',
            label=template,
            color=color
        )

    ax.set_xlabel('Epoch', fontsize=14)
    ax.set_ylabel(model_name, fontsize=25)


def draw_template_base(template_type, deduped, dataset, name):
    models = MODELS

    if deduped:
        models = [(model + '-deduped') for model in models]

    base_success = [row[template_type] for row in BASE_RESULTS]

    # half_success = [row[template_type] for row in half_results]

    generated_success = [np.max(dataset.filter(lambda x:
                                               (x['base_model'] == model_name) and
                                               (x['finetune_type'] == FINETUNE_TYPE) and
                                               (x['template_type'] == template_type))['success']) for model_name in models]

    bests = dataset.filter(lambda x:
                           ((x['success'], x['base_model']) in zip(generated_success, models)) and
                           (x['finetune_type'] == FINETUNE_TYPE) and
                           (x['template_type'] == template_type))

    for i in range(len(bests)):
        print(bests[i])

    if INCLUCE_ORIGINAL:
        original_success = [np.max(dataset.filter(lambda x:
                                                  (x['base_model'] == model_name) and
                                                  (x['finetune_type'] == 'original') and
                                                  (x['template_type'] == template_type))['success']) for model_name in models]

    fig, ax = plt.subplots(1, 1, figsize=(24, 16))

    x = np.arange(len(models))
    width = (0.25 if INCLUCE_ORIGINAL else 0.32)

    ax.bar(
        (x - width if INCLUCE_ORIGINAL else x - width/2),
        base_success,
        width,
        color='skyblue',
        label='Base'
    )

    # ax.bar(
    #     (x if INCLUCE_ORIGINAL else x + width/2),
    #     half_success,
    #     width,
    #     color='red',
    #     label='Initial'
    # )

    ax.bar(
        (x if INCLUCE_ORIGINAL else x + width/2),
        generated_success,
        width,
        color='blue',
        label='Generated'
    )

    if INCLUCE_ORIGINAL:
        ax.bar(
            x + width,
            original_success,
            width,
            color='red',
            label='original'
        )

    ax.set_xticks(x, [re.search('[^-]*-(.*)$', model, re.MULTILINE).group(1)
                  for model in models], fontsize=(55 if deduped else 120))

    ax.set_ylabel("Template " + template_type, fontsize=100)
    # ax.set_title("Template " + template_type, fontsize=75)
    ax.set_yticklabels([int(element)
                       for element in ax.get_yticks()], fontsize=80)

    handles = []
    handles.append(mpatches.Patch(color='skyblue', label='Base Model'))
    handles.append(mpatches.Patch(color='blue', label='Generated Data'))
    if (INCLUCE_ORIGINAL):
        handles.append(mpatches.Patch(color='red', label='Original Data'))

    # Add a single legend to the figure
    fig.legend(handles=handles, loc='upper center', ncol=(
        3 if INCLUCE_ORIGINAL else 2), fontsize=(45 if INCLUCE_ORIGINAL else 55))

    fig.savefig(name + '.pdf')
    fig.savefig(name + '.png')


def prepare_dataset(raw_data):
    # Create a Dataset object
    print(raw_data)
    df = pd.DataFrame(raw_data)

    df.drop_duplicates(
        subset=['base_model', 'finetune_type', 'weight_decay', 'learning_rate',
                'checkpoint', 'template_type', 'temperature', 'top-k'],
        inplace=True
    )

    return Dataset.from_pandas(df)


if __name__ == "__main__":
    results = []
    for base_path in BASE_PATHES:
        for root, _, files in os.walk(base_path):
            for file in files:
                if file == "results_purified.jsonl":
                    file_path = os.path.join(HOME_DIR, root, file)
                    for result in parse_results(file_path):
                        results.append(result)

    print(f"results array size: {len(results)}")
    # column_names = ['base_model', 'finetune_type', 'weight_decay', 'learning_rate',
                    # 'checkpoint', 'template_type', 'temperature', 'top-k', 'success']

    # Convert list of lists to list of dictionaries
    # data_dict = {col: [row[i] for row in results]
                #  for i, col in enumerate(column_names)}

    dataset = prepare_dataset(results)

    for template in ['A', 'B', 'C', 'D']:
        print(template)
        draw_template_base(
            template, False, dataset, f'/home/c01mili/CISPA-projects/llm_ftsec-2024/atilla/enron/bash_scripts/results/opt-dora/new_graphs/{template}')
