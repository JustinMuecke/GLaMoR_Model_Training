from transformers import AutoTokenizer, Llama4ForConditionalGeneration, AutoModelForCausalLM
import torch
import pandas as pd
import ast
from typing import List, Dict
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score
torch.manual_seed(42)
import wandb



token = os.getenv("HF_Token")



def load_data():
    train_df = pd.read_csv("../data/train_data.csv", header=0)
    eval_df = pd.read_csv("../data/eval_data.csv", header=0)
    test_df = pd.read_csv("../data/test_data.csv", header=0)

    train_graphs = [ast.literal_eval(triples) for triples in train_df["body"]]
    eval_graphs = [ast.literal_eval(triples)for triples in eval_df["body"]]
    test_graphs = [ast.literal_eval(triples)for triples in test_df["body"]]

    graph_dict= {"train" : train_graphs, "eval":eval_graphs, "test":test_graphs}

    label_2_index = {"Inconsistent" : 1, "Consistent" : 0}
    label_dict= {"train" : train_df["consistency"].map(label_2_index), "eval" : eval_df["consistency"].map(label_2_index), "test" : test_df["consistency"].map(label_2_index)}
    return graph_dict, label_dict, label_2_index

def graph_to_t5_input(triples) -> str:
    return " ".join([f"{s} {p} {o}." for s, p, o in triples])

def graphs_to_t5_format(triples , labels: List[str]) -> List[Dict[str, str]]:
    return [
        {
            "input_ids": graph_to_t5_input(graph),
            "label": label
        }
        for graph, label in tqdm(zip(triples, labels), total=len(triples))
    ]

def tokenize_data(tokenizer, data, max_input_length=4096):
    input_texts = [d["input_ids"] for d in data]
    target_labels = [d["label"] for d in data]

    # Initialize lists to store the tokenized sequences
    model_inputs = {
        "input_ids": [],
        "attention_mask": []
    }
    for text in tqdm(input_texts, total=len(input_texts)):
        encoding = tokenizer(
            text,
            padding="max_length",  # Ensure all sequences are padded to max_input_length
            max_length=max_input_length,
            truncation=True,
            return_tensors="pt"
        )
        model_inputs["input_ids"].append(encoding.input_ids)
        model_inputs["attention_mask"].append(encoding.attention_mask)

    labels = torch.tensor(target_labels, dtype=torch.long)
    model_inputs["labels"] = labels
    return model_inputs



def main():
    model_id = "meta-llama/Meta-Llama-3-8B"
#    model_id = "deepseek-ai/deepseek-r1"
#    moder_id = "Qwen/Qwen1.5-0.5B"
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=token)
    model = AutoModelForCausalLM.from_pretrained(model_id , device_map="auto", token=token)

# Your dataset
    graphs, labels, _ = load_data()

# Convert input to triples + label dict
    data = {
        "test": graphs_to_t5_format(graphs["test"], labels["test"])
    }

# Prefix and postfix for prompt
    prefix = "Determine whether the following text describes a consistent ontology. Each sentence represents a single triple.\n\n"
    postfix = "\n\nAnswer with 0 if it is consistent and 1 if it is inconsistent.\nAnswer:"

# Ground truth and predictions
    y_true = []
    y_pred = []
    model.config.pad_token_id = model.config.eos_token_id
    # Loop over data
    for example in tqdm(data["test"], total=len(data["test"])):
        input_text = prefix + example["input_ids"] + postfix
        inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

        model.eval()
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=2,
                do_sample=False,
                top_k=0,
                temperature=0,
                eos_token_id=tokenizer.eos_token_id
            )

        decoded_output = tokenizer.decode(output[0], skip_special_tokens=True)

        prediction = decoded_output.split("Answer:")[-1].strip()
        #Make sure only 0 or 1 are used
        if prediction not in ["0", "1"]:
            print("Warning: Unexpected prediction:", prediction)
            with open("errors.txt", "a") as f:
                f.write(decoded_output)
                f.write("\n\n")
            continue

        y_true.append(example["label"])
        y_pred.append(int(prediction))

# Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    wandb.log({
        "test/accuracy":accuracy, "test/precision":precision, "test/recall":recall
    })


if __name__ == "__main__":
    for i in range(5):
        wandb.init(project="Qwen", name="LLama-test")
        main()
        wandb.finish()

