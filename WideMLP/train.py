import os 
import pandas as pd
import numpy as np
import torch.nn as nn
import torch
from widemlp import WideMLP
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data import random_split
from sklearn.metrics import precision_score, recall_score


def train_epoch(train_loader: DataLoader, model : nn.Model, optimizer, loss_fn, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for inputs, labels in train_loader:
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
        for inputs, labels in val_loader:
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

def train_model(train_loader : DataLoader, val_loader : DataLoader, test_data : DataLoader, device):

    model = WideMLP()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()    
    for epoch in range(50):    
        loss, accuracy = train_epoch(train_loader, model, optimizer, loss_fn, device)
        loss, accuracy, recall, precision = val_epoch(val_loader, model, loss_fn, device)


def main(path : str, device):
    embeddings = {}
    for file in os.listdir(path):
        if(".npy" in file):
            array = np.load(path+file)
            embeddings[file] = array

    dataset = pd.read_csv("../../GLaMoR/data/filtered_dataset.csv", header = 0)
    x = [embeddings.get(filename.split(".")[0]+".npy", np.full((100,), np.NaN)) for filename in dataset["file_name"].values.tolist()]
    # Convert list of arrays into a DataFrame (each row corresponds to one array)
    data = pd.DataFrame(x, columns=[f"feature_{i}" for i in range(100)])  # Adjust number of features as needed
    data = data.dropna(subset=[f"feature_{i}" for i in range(100)])


    


    y_tensor = torch.tensor(dataset['consistency'].values, dtype=torch.long)
    x_tensor = torch.tensor(data.iloc[:, :100].values, dtype=torch.float32)
    dataset = TensorDataset(x_tensor, y_tensor)

    train_size = int(len(data) * 0.7)
    val_size = int(len(data) * 0.15)
    test_size = int(len(data) * 0.15)

    train_set, val_set, test_set = random_split(dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_set, batch_size = 8)
    val_loader = DataLoader(val_set, batch_size=8)
    test_loader = DataLoader(test_set, batch_size=8)


    model = train_model(train_loader, val_loader, test_loader, device)


if __name__ == "__main__":
    device="cpu"
    main(device)