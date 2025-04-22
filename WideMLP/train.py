import os 
import pandas as pd
import numpy as np
import torch.nn as nn
import torch
from widemlp import WideMLP
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data import random_split
from sklearn.metrics import precision_score, recall_score
from datetime import datetime
import wandb
from tqdm import tqdm

def train_epoch(train_loader: DataLoader, model, optimizer, loss_fn, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for inputs, labels in tqdm(train_loader):
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)

        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # Compute predictions and accuracy
        preds = torch.argmax(outputs, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    avg_loss = total_loss / len(train_loader)
    accuracy = correct / total

    return avg_loss, accuracy


def val_epoch(val_loader, model, loss_fn, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in tqdm(val_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            total_loss += loss.item()

            preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(val_loader)
    accuracy = (np.array(all_preds) == np.array(all_labels)).mean()
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)

    return avg_loss, accuracy, precision, recall

def train_model(train_loader : DataLoader, val_loader : DataLoader, test_loader : DataLoader, device, lr, wd, do):

    model = WideMLP(dropout_rate = do).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay = wd)
    loss_fn = nn.CrossEntropyLoss()    
    
    best_epoch = 0
    best_eval_accuracy = 0
    best_eval_loss = float('inf')
    best_test_accuracy = 0
    best_test_loss = float('inf')
    stopped_early = False
    train_start = datetime.now()

    for epoch in range(50):    
        train_loss, train_accuracy = train_epoch(train_loader, model, optimizer, loss_fn, device)
        eval_loss, eval_accuracy, eval_precision, eval_recall = val_epoch(val_loader, model, loss_fn, device)
        
        if eval_loss < best_eval_loss:
            best_epoch = epoch
            best_eval_accuracy = eval_accuracy
            best_eval_loss = eval_loss
            best_eval_accuracy = eval_accuracy
            best_eval_loss = eval_loss

        wandb.log(
            {
                "epoch": epoch,
                "best_epoch": best_epoch,
                "stopped_early": float(stopped_early),
                "train/accuracy": train_accuracy, "train/loss": train_loss,
                "eval/accuracy": eval_accuracy, "eval/precision" : eval_precision, "eval/recall" : eval_recall, "eval/loss": eval_loss, 'eval/best_accuracy': best_eval_accuracy, 'dev/best_loss': best_eval_loss,
            }
        )
        last_epoch = epoch
        if epoch - best_epoch >= 5:
            stopped_early = True
            break

    for epoch in range(last_epoch+1, 50):
        wandb.log(
            {
                "epoch": epoch,
                "best_epoch": best_epoch,
                "stopped_early": float(stopped_early),
                "train/accuracy": train_accuracy, "train/loss": train_loss,
                "eval/accuracy": eval_accuracy, "eval/precision" : eval_precision, "eval/recall" : eval_recall, "eval/loss": eval_loss,
            }
        )
    train_end = datetime.now()

    wandb.log({"time/training" : str(train_end - train_start)})
    inf_start = datetime.now()
    test_loss, test_accuracy, test_precision, test_recall = val_epoch(test_loader, model, loss_fn, device)
    inf_end = datetime.now()
    wandb.log({"time/inference" : str(inf_end - inf_start)})

def main(path : str, device, lr, wd, do):
    embeddings = {}
    for file in os.listdir(path):
        if(".npy" in file):
            array = np.load(path+file)
            embeddings[file] = array

    dataset = pd.read_csv("../data/filtered_dataset.csv", header = 0)
    x = [embeddings.get(filename.split(".")[0]+".npy", np.full((100,), np.NaN)) for filename in dataset["file_name"].values.tolist()]
    # Convert list of arrays into a DataFrame (each row corresponds to one array)
    data = pd.DataFrame(x, columns=[f"feature_{i}" for i in range(100)])  # Adjust number of features as needed

    valid_rows = ~data.isnull().any(axis=1)
    data = data[valid_rows]
    labels = dataset.loc[valid_rows, 'consistency'] 


    


    y_tensor = torch.tensor(labels.values, dtype=torch.long)
    x_tensor = torch.tensor(data.values, dtype=torch.float32)
    print(y_tensor.shape)
    print(x_tensor.shape)
    dataset = TensorDataset(x_tensor, y_tensor)

    train_size = int(len(data) * 0.7)
    val_size = int(len(data) * 0.15)
    test_size = len(data) - train_size - val_size

    train_set, val_set, test_set = random_split(dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_set, batch_size = 8)
    val_loader = DataLoader(val_set, batch_size=8)
    test_loader = DataLoader(test_set, batch_size=8)


    model = train_model(train_loader, val_loader, test_loader, device, lr, wd, do)


if __name__ == "__main__":
    dataset = "../data/embeddings/"
    device="cuda:3"
#    learning_rates = [1e-4, 1e-5, 5e-5]
#    weight_decays = [0, 1e-4, 1e-5]
#    dropouts = [0.1, 0.3, 0.5]
    lr = 5e-5
    wd = 0
    dropout = 0.5
    run_id = 0

    for i in range(4):
        name = f"WideMLP-lr{lr}-wd{wd}-do{dropout}"
        wandb_run = wandb.init(
            project="WideMLP",
            name=name,
            config={
                "learning_rate": lr,
                "weight_decay": wd,
                "dropout": dropout,
                "device": device
            }
        )

        main(
            dataset,
            device,
            lr,
            wd,
            dropout,
        )

        wandb_run.finish()
        run_id += 1
