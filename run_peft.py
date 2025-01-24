from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model
from datasets import load_dataset, Dataset
from accelerate import PartialState
from trl import SFTTrainer
import os
import re
import sys
import argparse
import torch

RESULTS_DIR = "./"
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['WORLD_SIZE'] = '1'
os.environ['MASTER_PORT'] = '1234'

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", help="model that will be instruction-tuned",
                        default="EleutherAI/pythia-1.4b")
    parser.add_argument("-i", "--input", help="training dataset path",
                        default="./data/enron.jsonl")
    parser.add_argument("-pdbs", "--per_device_batch_size", help="",
                        type=int, default=1)
    parser.add_argument("-gas", "--gradient_accumulation_steps", help="",
                        type=int, default=8)
    parser.add_argument("-w", "--warmup_steps", help="",
                        type=int, default=0)
    parser.add_argument("-e", "--epochs", help="",
                        type=int, default=4)
    parser.add_argument("-lr", "--learning_rate", help="",
                        type=float, default=2e-6)
    parser.add_argument("-l", "--logging_steps", help="",
                        type=int, default=1)
    parser.add_argument("-opt", "--optimizer", help="",
                        default="adamw_torch")
    parser.add_argument("-wd", "--weight_decay", help="",
                        type=float, default=0)
    parser.add_argument("-s", "--scheduler", help="",
                        default="linear")
    parser.add_argument("-dt", "--distribution_type", help="It can be ddp, mp or fsdp",
                        type=str, default="ddp")
    parser.add_argument("-pt", "--peft_type", help="It can be lora, dora and pissa",
                        type=str, default="lora")

    return parser.parse_args()


def get_output_dir(model_name, wd, lr, is_original):
    """
    Generates an output directory path based on model name, weight decay, learning rate, and originality status.

    Args:
        model_name (str): Name of the model.
        wd (float): Weight decay.
        lr (float): Learning rate.
        is_original (bool): Flag indicating whether the model is original.

    Returns:
        str: The path to the output directory.
    """
    # Extract the base name of the model
    output_directory = re.search(r'[^/]+$', model_name, re.MULTILINE).group(0)

    # TODO: fix this
    # Append additional directory structure based on the is_original flag
    if is_original:
        output_directory = os.path.join(output_directory, "self-generated-dora-" +
                                        re.search(r'/(pythia[^/]*|opt-[^/]*)', args.input).group(1))
    else:
        output_directory = os.path.join(output_directory, "original-dora")

    # Add weight decay and learning rate information
    output_directory = os.path.join(output_directory, f"{wd}_{lr}")

    # Create the results directory using os.path
    results_dir = os.path.join(RESULTS_DIR, output_directory)
    os.makedirs(results_dir, exist_ok=True)

    return results_dir


def formatting_prompts_func(example, eos_token):
    exampleGenerationPrompt = "{prompt}{continuation}"

    example["text"] = exampleGenerationPrompt.format(
        prompt=example["prompt"],
        continuation=example["continuation"]
    ) + eos_token

    return example


def get_dataset(input):
    try:
        dataset = load_dataset("json", data_files=input, split="train")
    except:
        dataset = Dataset.from_file(input)

    dataset = dataset.select(range(len(dataset) - 2220, len(dataset)))

    return dataset


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}")


def format_rate(number):
    return format(float(number), ".0e")


def get_device_map(distri_type):
    if distri_type == "ddp":
        device_index = PartialState().process_index
        return {"": device_index}

    if distri_type == "mp":
        return "auto"

    return None


def get_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token

    return tokenizer


def tokenize_function(item, tokenizer):
    item = tokenizer(
        item["text"],
        truncation=True,
        padding="max_length",
        max_length=2048,
    )
    item["labels"] = item["input_ids"]
    return item


def prepare_dataset(dataset, tokenizer):
    eos_token = tokenizer.eos_token
    # If no text column is provided.
    if 'text' not in dataset.column_names:
        dataset = dataset.map(
            lambda item: formatting_prompts_func(item, eos_token))

    dataset = dataset.map(
        lambda item: tokenize_function(item, tokenizer),
        remove_columns=dataset.column_names,
        batched=True,
    )

    return dataset


def get_pissa():
    return LoraConfig(
        r=64,
        lora_alpha=16,
        target_modules="all-linear",
        # Since the component of the PiSSA adapter are the principal singular values and vectors, dropout should be set to 0 to avoid random discarding.
        lora_dropout=0,
        bias="none",
        task_type="CAUSAL_LM",
        init_lora_weights="pissa",
    )


def get_dora():
    return LoraConfig(
        r=64,
        lora_alpha=16,
        target_modules="all-linear",
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        use_dora=True
    )


def get_lora():
    return LoraConfig(
        r=64,
        lora_alpha=16,
        target_modules="all-linear",
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )


def get_peft_config(peft_type):
    if peft_type == "pissa":
        return get_pissa()

    if peft_type == "dora":
        return get_dora()

    if peft_type == "lora":
        return get_lora()

    raise ValueError(
        "Unrecognized value for peft model. We only support lora, dora and pissa")


def get_training_arguments(args, output_directory):
    return TrainingArguments(
        per_device_train_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_steps=args.warmup_steps,
        num_train_epochs=args.epochs,
        save_strategy="epoch",
        learning_rate=args.learning_rate,
        logging_steps=args.logging_steps,
        optim=args.optimizer,
        fp16=True,
        weight_decay=args.weight_decay,
        lr_scheduler_type=args.scheduler,
        seed=3407,
        output_dir=output_directory,
    )


def run_peft(args):
    wd = format_rate(args.weight_decay)
    lr = format_rate(args.learning_rate)

    output_directory = get_output_dir(
        args.model, wd, lr, 'continuations' in args.input)

    # TODO: remove this. Add wandb's logging.
    sys.stdout = open(output_directory + "trajectory.txt",
                      "w")  # override print

    dataset = get_dataset(args.input)
    device_map = get_device_map(args.distribution_type)

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=False,
        cache_dir="",
        device_map=device_map,
        # device_map="auto",
        use_cache=False,
    )

    if args.distribution_type != "fsdp":
        model.gradient_checkpointing_disable()

    tokenizer = get_tokenizer(args.model)
    dataset = prepare_dataset(dataset, tokenizer)

    peft_config = get_peft_config(args.peft_type)
    training_arguments = get_training_arguments(args, output_directory)

    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    if args.peft_type == "pissa":
        model.peft_config["default"].init_lora_weights = True

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=2048,
        dataset_num_proc=2,
        packing=False,
        args=training_arguments
    )

    # @title Show current memory stats
    gpu_stats = torch.cuda.get_device_properties(0)
    start_gpu_memory = round(
        torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
    print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
    print(f"{start_gpu_memory} GB of memory reserved.")

    # TODO: fix fsdp's model saivng process.

    trainer_stats = trainer.train()

    # @title Show final memory and time stats
    used_memory = round(torch.cuda.max_memory_reserved() /
                        1024 / 1024 / 1024, 3)
    used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
    used_percentage = round(used_memory / max_memory*100, 3)
    lora_percentage = round(used_memory_for_lora/max_memory*100, 3)
    print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
    print(f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training.")
    print(f"Peak reserved memory = {used_memory} GB.")
    print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
    print(f"Peak reserved memory % of max memory = {used_percentage} %.")
    print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")

    print("fine tuned model is saved at " + output_directory)


if __name__ == '__main__':
    args = parse_args()

    run_peft(args)
