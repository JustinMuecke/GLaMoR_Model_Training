from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree, softmax, remove_self_loops
from torch_geometric.nn import GCNConv, global_max_pool, Linear, global_mean_pool
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

import math
import numpy as np
import itertools
import random
import copy
from tqdm import tqdm

import site
import sys

site.addsitedir('../lib')  # Always appends to end
import gc

seed = 42  # seed for the random operations


# for link prediction
class EdgeDecoder(torch.nn.Module):
    def __init__(self, hidden_channels, device):
        """
        self: itself.
        hidden_channels (int): Dimension of the hidden dimensions.

        Return: EdgeDecoder(torch.nn.Module).
        """
        super().__init__()
        self.device = device
        self.lin1 = Linear(2 * hidden_channels, hidden_channels).to(device)
        self.lin2 = Linear(hidden_channels, 1).to(device)
        self.bn = torch.nn.BatchNorm1d(hidden_channels).to(device)
        self.simC = torch.nn.CosineSimilarity(dim=1, eps=1e-6)

    def forward(self, x, edge_index, sim=False):
        z = torch.zeros(len(edge_index), 2 * len(x[0])).to(self.device)
        i = 0
        for e1, e2 in edge_index:
            z[i] = torch.cat(([x[e1], x[e2]]), dim=-1)
            i += 1

        if sim:
            z = self.simC(x[edge_index][:, 0], x[edge_index][:, 1]).reshape(len(z), 1)
        else:
            z = self.lin1(z)
            z = self.bn(z)
            z = F.relu(z)
            z = self.lin2(z)
        return z


# For reasoning. Similar to EdgeDecoder, except that the classification is not necessarily binary.
class EdgeReasoner(torch.nn.Module):
    def __init__(self, hidden_channels, output_channels, device):
        """
        self: itself.
        hidden_channels (int): Dimension of the hidden dimensions.
        output_channel (int): Number of classes for the reasoning

        Return: EdgeReasoner(torch.nn.Module).
        """

        super().__init__()
        self.device = device
        self.lin1 = Linear(2 * hidden_channels, hidden_channels).to(device)
        self.lin2 = Linear(hidden_channels, output_channels).to(device)
        self.bn = torch.nn.BatchNorm1d(hidden_channels).to(device)

    # Forwading method of the EdgeReasoner
    def forward(self, x, edge_index):
        """
        self: itself.
        x (torch.tensor): The features of the nodes.
        edge_index (torch.tensor): Edges for classifcation.
        sim(Boolean): If True only similiarity is used between two nodes.

        Return: Predictions for the edges.
        """

        z = torch.zeros(len(edge_index), 2 * len(x[0])).to(self.device)
        i = 0
        for e1, e2 in edge_index:
            z[i] = torch.cat(([x[e1], x[e2]]), dim=-1)
            i += 1

        z = self.lin1(z)
        z = self.bn(z)
        z = F.relu(z)
        z = self.lin2(z)
        return z

# GNNs
# GNNs
class GNN(torch.nn.Module):
    def __init__(
            self,
            in_channels,
            hidden_channels,
            out_channels,
            n_layers,
            device,
            model="GCN",
            dropout=0.5,
            out=True
    ):
        """
        self: itself.
        in_channels (int): Length of the input vectors (features).
        hidden_channels (int): Dimension of the hidden dimensions.
        out_channesl (int): Number of classes.
        n_layers (int): Number of layers of the GNN
        device (torch.device): device used for the calculations, e.g., cpu or cuda.
        model (String): Determines the convolution layer type of the GNN.
        dropout (float): Between 0 and 1.
        out (boolean): Determines if the embedding or the predictions are returned by forward().

        Return: GNN(torch.nn.Module)
        """

        super().__init__()
        self.cons = torch.nn.ModuleList().to(device)
        if model == "GIN":
            linear = Linear(in_channels, hidden_channels).to(device)
            self.cons.append(GINConv(linear, train_eps=True)).to(device)
        elif model == "GAT":
            self.cons.append(GATConv(in_channels, hidden_channels)).to(device)
        else:
            self.cons.append(GCNConv(in_channels, hidden_channels)).to(device)
        for _ in range(n_layers - 1):
            if model == "GIN":
                linear = Linear(hidden_channels, hidden_channels).to(device)
                self.cons.append(GINConv(linear, train_eps=True)).to(device)
            elif model == "GAT":
                self.cons.append(GATConv(hidden_channels, hidden_channels)).to(device)
            else:
                self.cons.append(GCNConv(hidden_channels, hidden_channels)).to(device)
        self.bn = torch.nn.BatchNorm1d(hidden_channels).to(device)
        self.dropout = dropout
        self.training = False
        self.linear = Linear(hidden_channels, out_channels).to(device)
        self.edge_decoder = EdgeDecoder(hidden_channels, device)
        self.edge_reasoner = EdgeReasoner(hidden_channels, out_channels, device)
        self.model = model
        self.out = out  # if it should return a output
        self.device = device

    def forward(self, x, edge_index, mask, task, features=None):
        """
        self: itself.
        x (torch.tensor): The features of the nodes.
        edge_index (torch.tensor): The edge  of the graph.
        mask (torch.tensor): The mask for the target nodes.
        task (String): Determines the current task: "node" (node classification), "link" or "linkS" (link prediction), "graph" (graph classification), "reason" (reasoning).

        Return: Return: The labels of the target nodes.
        """
        return x, 0
        for i in range(len(self.cons)):
            if i == 0:
                if features is None:
                    x = self.cons[i](x, edge_index)
                else:
                    x = self.cons[i](x, edge_index, features)
            else:
                if features is None:
                    x = self.cons[i](x, edge_index)
                else:
                    x = self.cons[i](x, edge_index, features)
            x = self.bn(x)
            if i < len(self.cons) - 1:
                x = F.relu(x)
                if self.training:
                    x = F.dropout(x, p=self.dropout)

        Y = None
        if self.out:
            if task == "node":
                Y = self.linear(x)[mask]
                Y = torch.softmax(Y, dim=-1)
                # Y = torch.nn.functional.softmax(Y)
            if task == "link":
                Y = self.edge_decoder(x, mask).to(self.device)
                Y = torch.sigmoid(Y)
            if task == "linkS":
                Y = self.edge_decoder(x, mask, True).to(self.device)
                Y = torch.sigmoid(Y)
                # print(torch.sigmoid(Y))
            if task == "graph":
                x = global_mean_pool(x, mask).to(self.device)  # mask is batch!!!
                Y = self.linear(x)
                Y = torch.sigmoid(Y)
            if task == "reason":
                Y = self.edge_reasoner(x, mask).to(self.device)
                Y = torch.softmax(Y, dim=-1)
                # Y = torch.nn.functional.softmax(Y)
        else:
            Y = x
        return Y, 0

        # (un-)freeze the parameters of the model
        def freeze(self, freeze=False):
            """
            freeze (boolean): Determines if gradient descent is possible for the parameters of the model

            Return: None
            """

            for p in self.parameters():
                p.requires_grad = not freeze


# getPrompts is used the get the prompts/examples for node classification
def getPrompts(y, ways, num_classes):
    """
    y (torch.Tensor): The labels of the n nodes from 0 to n-1.
    ways (int): Number of examples/prompts per class.
    num_classes (int): Number of the c different classes from 0 to c-1.

    Return: A dictionary “prompts” containing each class's list of prompts.
    """

    members = {}
    labels = y
    nodes = np.array(range(len(y)))
    for i in range(len(labels)):
        if int(labels[i]) in members:
            members[int(labels[i])].append(int(nodes[i]))
        else:
            members[int(labels[i])] = [int(nodes[i])]
    random.seed(seed)
    prompts = [[-1 for col in range(ways)] for row in range(num_classes)]
    for i in range(num_classes):
        j = 0
        while j < ways:
            if len(members[i]) < ways:
                print("ways too big. Not enough members of the class")
                return None
            prompt = random.choice(members[i])
            if prompt not in prompts[i]:
                prompts[i][j] = prompt
                j += 1
    return prompts


# getPromptsE is used the get the prompts/examples for link prediction
def getPromptsE(y, edges, ways):
    """
    y (torch.Tensor): The labels of the n edges from 0 to n-1.
    edge(torch.Tensor): The edges for the respective labels in y
    ways (int): Number of examples/prompts per class.

    Return: An array “prompts” containing each class's list of prompts. The classes are edges and no edges.
    """

    members = {}
    members[1] = edges[(y == 1).nonzero().to(edges.device)]
    members[0] = edges[(y == 0).nonzero().to(edges.device)]
    random.seed(seed)

    prompts = [[torch.tensor([-1, -1]) for col in range(ways)] for row in range(2)]
    for i in range(2):
        j = 0
        while j < ways:
            if len(members[i]) < ways:
                print("ways too big. Not enough members of the class")
                return None
            prompt = random.choice(members[i])[0]
            if sum(list(list(prompt) == list(k) for k in prompts[i])) == 0:
                prompts[i][j] = prompt
                j += 1
    return prompts


# getPromptsR is used the get the prompts/examples for reasoning (predicting edge labels)
def getPromptsR(y, edges, ways, num_classes):
    """
    y (torch.Tensor): The labels of the n edges from 0 to n-1.
    edge(torch.Tensor): The edges for the respective labels in y.
    ways (int): Number of examples/prompts per class.
    num_classes (int): Number of the c different classes from 0 to c-1.

    Return: An array “prompts” containing each class's list of prompts.
    """

    members = {}
    labels = y
    nodes = np.array(range(len(y)))
    for i in range(len(labels)):
        if int(labels[i]) in members:
            members[int(labels[i])].append(edges[i])
        else:
            members[int(labels[i])] = [edges[i]]
    random.seed(seed)
    prompts = [[torch.tensor([-1, -1]) for col in range(ways)] for row in range(num_classes)]
    for i in range(num_classes):
        j = 0
        while j < ways:
            if len(members[i]) < ways:
                print("ways too big. Not enough members of the class")
                return None
            prompt = random.choice(members[i])
            notM = True
            for k in prompts[i]:
                if k[0] == prompt[0] and k[1] == prompt[1]:
                    notM = False
            if notM:
                prompts[i][j] = prompt
                j += 1
    return prompts


# getPromptsGB is used the get the prompts/examples for binary graph classification
def getPromptsGB(y, ways):
    """
    y (torch.Tensor): The labels of the n graphs from 0 to n-1.
    ways (int): Number of examples/prompts per class.

    Return: A dictionary “prompts” containing each class's list of prompts.
    """

    members = {}
    labels = y
    nodes = np.array(range(len(y)))
    num_classes = 2
    for i in range(len(labels)):
        if int(labels[i]) in members:
            members[int(labels[i])].append(int(nodes[i]))
        else:
            members[int(labels[i])] = [int(nodes[i])]
    random.seed(seed)
    prompts = [[-1 for col in range(ways)] for row in range(num_classes)]
    for i in range(num_classes):
        j = 0
        while j < ways:
            if len(members[i]) < ways:
                print("ways too big. Not enough members of the class")
                return None
            prompt = random.choice(members[i])
            if prompt not in prompts[i]:
                prompts[i][j] = prompt
                j += 1
    return prompts


# getPromptsGB is used the get the prompts/examples for multi-way binary graph classification
def getPromptsGBM(y, ways):
    """
    y (torch.Tensor): The labels of the n graphs from 0 to n-1.
    ways (int): Number of examples/prompts per class.

    Return: A dictionary “prompts” containing each class's list of prompts.
    """

    members = {}
    labels = y
    nodes = np.array(range(len(y)))
    num_classes = 2
    classes = len(labels[0])
    for i in tqdm(range(len(labels))):
        for k in range(classes):
            if labels[i][k] != -1:
                if (k, labels[i][k]) in members:
                    members[k, int(labels[i][k])].append(int(nodes[i]))
                else:
                    members[k, int(labels[i][k])] = [int(nodes[i])]
        random.seed(seed)
    prompts = [[[-1 for col in range(ways)] for cn in range(num_classes)] for c in range(classes)]
    for i, k in members.keys():
        j = 0
        while j < ways:
            if len(members[i, k]) < ways:
                print("ways too big. Not enough members of the class")
                return None
            prompt = random.choice(members[i, k])
            if prompt not in prompts[i][k]:
                prompts[i][k][j] = prompt
                j += 1
    return prompts


# train is used to train the PRODIGY model with the training set and it uses the validation set to get the best model with early stopping.
def train(model, maxQ, x, yAll, edge_index, prompt_mask, query_mask, prompts, numberQ, drop, mask, epochs, optimizer,
          task, val_x, val_y, val_edge, val_mask, patience, name):
    """
    model (Prodigy(torch.nn.Module)): The Prodigy model to train.
    maxQ (int): Maximal number of queries sets used for one loss and backpropagation step.
    x (torch.tensor): The features of the nodes.
    yAll (torch.tensor): The labels for the current task in the train set.
    edge_index (torch.tensor): The edges.
    prompt_mask (torch.tensor): The mask/batch for the prompts/examples.
    query_mask (torche.tensor): The mask/batch for the queries in the test set.
    prompts (dict): A dictionary containing each class's list of prompts.
    numberQ (int): The maximal number of queries in a run of Prodigy or the maximal number of queries in a query set.

    drop (float): It is between 0 and 1 and is the probability that a vertex is dropped.
    mask (float): It is between 0 and 1 and is the probability that the features of a vertex  are masked by setting them to 0.
    epochs (int): How many epochs the model is trained on the training set.
    optimizer (torch.optim.Adam): The optimizer.
    task (String): Determines the current task: "node" (node classification), "link" or "linkS" (link prediction), "graph" (graph classification), "reason" (reasoning).

    val_x (torch.tensor): The features of the nodes in the validation set.
    val_y (torch.tensor): The labels for the current task in the validation set.
    val_edge (torch.tensor): The edges in the validation set.
    val_mask (torche.tensor): The mask/batch for the queries in the validation set.
    patience (int): For the early stopping.
    name (String): The name used to save the model.

    Return: None
    """

    model.training = True
    queriesAll = None
    random.seed(seed)
    best_val_loss = float('inf')
    patience = patience
    best_epoch = 0
    patience_counter = 0
    best_modle = None

    # Training
    for i in tqdm(range(epochs)):
        # Preparing the queries using a random order for each epoch
        shuffle = list(range(0, len(yAll)))
        random.shuffle(shuffle)
        if task == "node":
            queriesAll = torch.range(0, len(yAll) - 1)[shuffle]
            queriesAll = queriesAll.split(numberQ)
        elif task == "link" or task == "linkS" or task == "reason":
            queriesAll = query_mask[shuffle]
            queriesAll = queriesAll.split(numberQ)
        elif task == "graph":
            queriesAll = torch.range(0, len(yAll) - 1)[shuffle]
            queriesAll = queriesAll.split(numberQ)

        yS = yAll[shuffle]
        xd = None
        if task == "graph":
            xd = [x[0].detach().clone(), x[1].detach().clone()]
        else:
            xd = x.detach().clone()

        # Starting one training run
        startQ = 0
        endQ = 0
        startY = 0
        endY = 0

        if i == 0:
            # early stopping using the validation loss
            lossV, accV = evaluate(model=model,
                                   x=val_x,
                                   y=val_y,
                                   edge_index=val_edge,
                                   prompt_mask=prompt_mask,
                                   query_mask=val_mask,
                                   prompts=prompts,
                                   numberQ=numberQ,
                                   task=task, metric=True)
            # print("V: ",lossV)
            if lossV < best_val_loss:
                best_val_loss = lossV
                patience_counter = 0
                torch.save(model.state_dict(), "models/" + name)
                best_epoch = i
            else:
                patience_counter += 1

        if patience_counter >= patience:
            break

        for k in range(0, len(queriesAll), maxQ):
            endQ += maxQ
            if endQ > len(queriesAll):
                endQ = len(queriesAll)
            queries = queriesAll[startQ:endQ]
            for q in queries:
                endY += len(q)
            y = yS[startY:endY]
            loss = 0
            if task == "graph" and len(y[0]) > 1:
                optimizer.zero_grad()
                pred = torch.zeros(y.size()).to(model.device)
                for j in range(len(y[0])):
                    queriesL = torch.range(0, len(y) - 1).to(model.device)
                    queries = queriesL[y[:, j] != -1].split(numberQ)
                    predP, lossP = model(xd, edge_index, prompt_mask, query_mask, queries, prompts[j], drop=drop,
                                         mask=mask, task=task)
                    loss += lossP
                    pred[y[:, j] != -1][:, j] = predP[:, 1]
                    loss += F.cross_entropy(predP[:, 1].to(model.device), torch.flatten(y[y[:, j] != -1][:, j]).float())
                loss.backward()
                optimizer.step()
            else:
                optimizer.zero_grad()
                pred, loss = model(xd, edge_index, prompt_mask, query_mask, queries, prompts, drop=drop, mask=mask,
                                   task=task)
                if task == "graph" or task == "link" or task == "linkS":
                    yp = torch.flatten(y)
                    loss += F.cross_entropy(pred, yp)
                else:
                    loss += F.cross_entropy(pred, y)
                loss.backward()
                optimizer.step()
                correct = 0
            startY = endY
            startQ = endQ
        # early stopping using the validation loss
        lossV, accV = evaluate(model=model,
                               x=val_x,
                               y=val_y,
                               edge_index=val_edge,
                               prompt_mask=prompt_mask,
                               query_mask=val_mask,
                               prompts=prompts,
                               numberQ=numberQ,
                               task=task, metric=True)
        # print("V: ",lossV)
        # print(loss)
        if lossV < best_val_loss:
            best_val_loss = lossV
            patience_counter = 0
            torch.save(model.state_dict(), "models/" + name)
            best_epoch = i + 1
        else:
            patience_counter += 1

        if patience_counter >= patience:
            break

    # load the best model of all runs
    model.load_state_dict(torch.load("models/" + name, weights_only=True))
    model.eval()  # best weights
    print("Best epoch: ", best_epoch)
    model.training = False
    # End of Training

    # Output: acc or roc auc for the training set
    if task == "node":
        queriesAll = torch.range(0, len(yAll) - 1).split(numberQ)
    elif task == "link" or task == "linkS" or task == "reason":
        queriesAll = query_mask[shuffle].split(numberQ)
    elif task == "graph":
        queriesAll = torch.range(0, len(yAll) - 1).split(numberQ)
    if task == "graph" and len(y[0]) > 1:
        pred = torch.zeros(yAll.size()).to(model.device)
        for i in range(len(yAll[0])):
            queriesL = torch.range(0, len(yAll) - 1).to(model.device)
            queries = queriesL[yAll[:, i] != -1].split(numberQ)
            predP, lossP = model(xd, edge_index, prompt_mask, query_mask, queries, prompts[i], drop=0, mask=0,
                                 task=task)  # )reuse = reuse)
            pred[yAll[:, i] != -1][:, i] = predP[:, 1]
    else:
        pred, loss = model(xd, edge_index, prompt_mask, query_mask, queriesAll, prompts, drop=0, mask=0, task=task)

    correct = 0
    if task == "node" or task == "reason":
        correct = (torch.max(pred, 1).indices.float() == yAll.float()).sum()
        acc = correct / len(yAll)
    elif task == "link" or task == "linkS":
        correct = (torch.round(torch.flatten(pred[:, 1])).long() == yAll.float()).sum()
        acc = correct / len(yAll)
        yp = torch.flatten(yAll).float()
        print(f'ROC AUC Train: {roc_auc_score(yp.cpu(), pred[:, 1].cpu().detach().numpy()):.4f}')
    elif task == "graph":
        if len(y[0]) > 1:
            correctsA = torch.flatten(torch.round(pred[yAll != -1])).long() == torch.flatten(yAll[yAll != -1]).float()
            correct = correctsA.sum()
            acc = correct / (len(torch.flatten(yAll[yAll != -1])))
            loss += F.cross_entropy(pred[yAll != -1].to(model.device),
                                    torch.flatten(yAll[yAll != -1]).float())  # should be binary
        else:
            yp = torch.flatten(yAll)
            correctsA = torch.flatten(torch.round(pred[:, 1])).long() == torch.flatten(yp).float()
            correct = correctsA.sum()
            acc = correct / (len(torch.flatten(yp)))
            loss += F.cross_entropy(pred, yp)

        if len(y[0]) > 1:
            minR = float(1)
            maxR = float(0)
            for i in range(len(y[0])):
                roc = roc_auc_score(yAll[:, i][yAll[:, i] != -1].cpu(),
                                    pred[:, i][yAll[:, i] != -1].cpu().detach().numpy())
                if roc < minR:
                    minR = roc
                if roc > maxR:
                    maxR = roc
            print(f'ROC AUC Val Max: {maxR:.4f}')
            print(f'ROC AUC Val Min: {minR:.4f}')
            print(
                f'ROC AUC Val All: {roc_auc_score(yAll[yAll != -1].cpu(), pred[yAll != -1].cpu().detach().numpy()):.4f}')
        else:
            yp = torch.flatten(yAll).float()
            print(f'ROC AUC Train: {roc_auc_score(yp.cpu(), pred[:, 1].cpu().detach().numpy()):.4f}')
    print(f'Accuracy Train: {acc:.4f}')


# evaluate is either used to validate the model on the validation set or to get its loss for the validation set depending on metric
def evaluate(model, x, y, edge_index, prompt_mask, query_mask, prompts, numberQ, task, metric=False):
    """
    model (Prodigy(torch.nn.Module)): The Prodigy model to validate.
    x (torch.tensor): The features of the nodes.
    y (torch.tensor): The labels for the current task in the validation set.
    edge_index (torch.tensor): The edges.
    prompt_mask (torch.tensor): The mask/batch for the prompts/examples.
    query_mask (torche.tensor): The mask/batch for the queries in the validation set.
    prompts (dict): A dictionary containing each class's list of prompts.

    numberQ (int): The maximal number of queries in a run of Prodigy or the maximal number of queries in a query set.
    task (String): Determines the current task: "node" (node classification), "link" or "linkS" (link prediction), "graph" (graph classification), "reason" (reasoning).
    metric (boolean): Returns loss for metric == True and for metric == False, it prints the acc or roc auc for the validation set.

    Return: loss (float) for metric == True.
    """

    # Preparing the queries
    queries = None
    if task == "node":
        queries = torch.range(0, len(y) - 1).split(numberQ)
    elif task == "link" or task == "linkS" or task == "reason":
        queries = query_mask.split(numberQ)
    elif task == "graph":
        queries = torch.range(0, len(y) - 1).split(numberQ)
    xd = None
    if task == "graph":
        xd = [x[0].detach().clone(), x[1].detach().clone()]
    else:
        xd = x.detach().clone()

    # Calculating the predictions of the model for the task and the validation set
    if task == "graph" and len(y[0]) > 1:
        pred = torch.zeros(y.size()).to(model.device)
        for i in range(len(y[0])):
            queriesL = torch.range(0, len(y) - 1).to(model.device)
            queries = queriesL[y[:, i] != -1].split(numberQ)
            predP, lossP = model(xd, edge_index, prompt_mask, query_mask, queries, prompts[i], drop=0, mask=0,
                                 task=task)  # )reuse = reuse)
            pred[y[:, i] != -1][:, i] = predP[:, 1]

    else:
        pred, loss = model(xd, edge_index, prompt_mask, query_mask, queries, prompts, drop=0, mask=0, task=task)
    correct = 0
    acc = 0
    loss = 0
    if task == "node" or task == "reason":
        correct = (torch.max(pred, 1).indices.float() == y.float()).sum()
        acc = correct / len(y)
        loss += F.cross_entropy(pred, y)
    elif task == "link" or task == "linkS":
        correct = (torch.round(torch.flatten(pred[:, 1])).long() == y.float()).sum()
        acc = correct / len(y)
        yp = torch.flatten(y)
        loss += F.cross_entropy(pred, yp)
    elif task == "graph":
        if len(y[0]) > 1:
            correctsA = torch.flatten(torch.round(pred[y != -1])).long() == torch.flatten(y[y != -1]).float()
            correct = correctsA.sum()
            acc = correct / (len(torch.flatten(y[y != -1])))
            loss += F.cross_entropy(pred[y != -1].to(model.device),
                                    torch.flatten(y[y != -1]).float())  # should be binary
        else:
            yp = torch.flatten(y)
            correctsA = torch.flatten(torch.round(pred[:, 1])).long() == torch.flatten(yp).float()
            correct = correctsA.sum()
            acc = correct / (len(torch.flatten(yp)))
            loss += F.cross_entropy(pred, yp)

    # return loss if metric == True
    if metric:
        return loss, acc
        # Output: acc or roc auc for the validation set
    if task == "link" or task == "linkS":
        print(f'ROC AUC Val: {roc_auc_score(yp.cpu(), pred[:, 1].cpu().detach().numpy()):.4f}')
    if task == "graph":
        if len(y[0]) > 1:
            minR = float(1)
            maxR = float(0)
            for i in range(len(y[0])):
                roc = roc_auc_score(y[:, i][y[:, i] != -1].cpu(), pred[:, i][y[:, i] != -1].cpu().detach().numpy())
                if roc < minR:
                    minR = roc
                if roc > maxR:
                    maxR = roc
            print(f'ROC AUC Val Max: {maxR:.4f}')
            print(f'ROC AUC Val Min: {minR:.4f}')
            print(f'ROC AUC Val All: {roc_auc_score(y[y != -1].cpu(), pred[y != -1].cpu().detach().numpy()):.4f}')
        else:
            yp = torch.flatten(y).float()
            print(f'ROC AUC Val: {roc_auc_score(yp.cpu(), pred[:, 1].cpu().detach().numpy()):.4f}')
    print(f'Accuracy Val: {acc:.4f}')


# test is used to test the model on the test set
def test(model, x, y, edge_index, prompt_mask, query_mask, prompts, numberQ, task):
    """
    model (Prodigy(torch.nn.Module)): The Prodigy model to validate.
    x (torch.tensor): The features of the nodes.
    y (torch.tensor): The labels for the current task in the test set.
    edge_index (torch.tensor): The edges.
    prompt_mask (torch.tensor): The mask/batch for the prompts/examples.
    query_mask (torche.tensor): The mask/batch for the queries in the test set.
    prompts (dict): A dictionary containing each class's list of prompts.
    numberQ (int): The maximal number of queries in a run of Prodigy or the maximal number of queries in a query set.

    task (String): Determines the current task: "node" (node classification), "link" or "linkS" (link prediction), "graph" (graph classification), "reason" (reasoning).

    Return: None
    """

    # Preparing the queries
    queries = None
    if task == "node":
        queries = torch.range(0, len(y) - 1).split(numberQ)
    elif task == "link" or task == "linkS" or task == "reason":
        queries = query_mask.split(numberQ)
    elif task == "graph":
        queries = torch.range(0, len(y) - 1).split(numberQ)

    xd = None
    if task == "graph":
        xd = [x[0].detach().clone(), x[1].detach().clone()]
    else:
        xd = x.detach().clone()

    # Calculating the predictions of the model for the task and the test set
    if task == "graph" and len(y[0]) > 1:
        pred = torch.zeros(y.size()).to(model.device)
        for i in range(len(y[0])):
            queriesL = torch.range(0, len(y) - 1).to(model.device)
            queries = queriesL[y[:, i] != -1].split(numberQ)
            predP, lossP = model(xd, edge_index, prompt_mask, query_mask, queries, prompts[i], drop=0, mask=0,
                                 task=task)  # )reuse = reuse)
            pred[y[:, i] != -1][:, i] = predP[:, 1]
    else:
        pred, loss = model(xd, edge_index, prompt_mask, query_mask, queries, prompts, drop=0, mask=0, task=task)
    correct = 0

    # Output: acc or roc auc for the test set
    if task == "node" or task == "reason":
        correct = (torch.max(pred, 1).indices.float() == y.float()).sum()
  
        _, predicted = torch.max(pred, 1)

        predicted = predicted.float()
        y = y.float()

        correct = (predicted == y).sum()
        accuracy = correct / len(y)

        TP = ((predicted == 1) & (y == 1)).sum()
        FP = ((predicted == 1) & (y == 0)).sum()
        FN = ((predicted == 0) & (y == 1)).sum()

        # Precision and Recall
        precision = TP / (TP + FP) if TP + FP > 0 else torch.tensor(0.0)
        recall = TP / (TP + FN) if TP + FN > 0 else torch.tensor(0.0)

        # Optionally print the values
        print(f"Accuracy: {accuracy.item()}")
        print(f"Precision: {precision.item()}")
        print(f"Recall: {recall.item()}")
    return accuracy.item(), precision.item(), recall.item()


# Prodigy
# Start code from https://github.com/snap-stanford/prodigy (used to implement Prodigy)
class MetaGNN(torch.nn.Module):
    def __init__(self, edge_attr_dim, emb_dim, heads=2, n_layers=1, dropout=0, aggr="add", has_final_back=False,
                 msg_pos_only=False, self_loops=True, batch_norm=True):
        super().__init__()
        self.num_gnn_layers = n_layers
        self.gnn_layers = torch.nn.ModuleList()
        self.msg_pos_only = msg_pos_only
        for i in range(self.num_gnn_layers):
            self.gnn_layers.append(
                MetaGNNLayer(emb_dim=emb_dim, heads=heads, edge_attr_dim=edge_attr_dim, dropout=dropout,
                             batch_norm=batch_norm))

        self.gnn_non_linear = torch.nn.ReLU()
        self.self_loops = self_loops

    def forward(self, x, edge_index, edge_attr):
        '''
        :param x: Feature matrix.
        :param edge_index: Edge index for the bipartite graph.
        :param edge_attr: Edge attributes for the bipartite graph.
        :return:
        '''
        if self.self_loops:
            num_nodes = x.size(0)
            edge_index, edge_attr = remove_self_loops(
                edge_index, edge_attr)
            edge_index, edge_attr = add_self_loops(
                edge_index, edge_attr, fill_value=torch.tensor([0, 1]).to(edge_attr.device),
                num_nodes=num_nodes)

        for i in range(self.num_gnn_layers):
            # hack for heterogeneous graph; should be fixed
            self.gnn_layers[i].training = self.training
            x = self.gnn_layers[i](x, edge_index, edge_attr=edge_attr)
            if i != self.num_gnn_layers - 1:
                x = self.gnn_non_linear(x)
        return x


# see "PRODIGY: Enabling In-context Learning Over Graphs" for the equation
class MetaGNNLayer(MessagePassing):
    """
    GAT gnn for bipartite graph.
    Args:
        edge_attr_dim (int): dimension of edge attributes (2 for metagraph).
        emb_dim (int): dimensionality of embeddings for nodes and edges.
        aggr (str): aggregation method. Default: "add".
    See https://arxiv.org/abs/1810.00826
    """

    def __init__(self, edge_attr_dim, emb_dim, heads=1, dropout=0, aggr="add", batch_norm=True):
        super(MetaGNNLayer, self).__init__()
        # k, q, v matrices, no bias for now
        self.heads = heads
        self.head_dim = emb_dim // heads

        self.mlp_kqv = torch.nn.Linear(emb_dim, 3 * emb_dim)
        self.emb_dim = emb_dim
        self.lin_edge = torch.nn.Linear(edge_attr_dim, emb_dim)
        self.att_mlp = torch.nn.Sequential(torch.nn.Linear(3 * self.head_dim, self.head_dim), torch.nn.ReLU(),
                                           torch.nn.Linear(self.head_dim, 1))
        self.out_proj = torch.nn.Linear(emb_dim, emb_dim)

        self.dropout = dropout
        self.aggr = aggr
        self.bn = torch.nn.Identity()
        if batch_norm:
            self.bn = torch.nn.BatchNorm1d(emb_dim)
        self.training = False

    def forward(self, x, edge_index, edge_attr=None):
        # print("meta: ",x.size())
        kqv_x = self.mlp_kqv(x)
        # runs the the layer (message passing))
        out = self.propagate(edge_index, x=kqv_x, edge_attr=edge_attr, size=None)
        out = F.dropout(out, p=self.dropout, training=self.training) + x
        out = self.bn(out)
        return out

    # rewrote the message-passing for the GNN!
    def message(self, x_j, x_i, edge_attr, index, ptr, size_i):
        H, E = self.heads, self.head_dim

        # compute query of target; k,v of source
        q = x_i[:, :self.emb_dim].reshape(-1, H, E)
        k = x_j[:, self.emb_dim: 2 * self.emb_dim].reshape(-1, H, E) / math.sqrt(E)
        v = x_j[:, 2 * self.emb_dim: 3 * self.emb_dim].reshape(-1, H, E)

        # apply linear layer to edge
        edge_attr = self.lin_edge(edge_attr)
        edge_attr = edge_attr.view(edge_attr.shape[0], H, E)

        # apply mlp to compute attention score
        alpha = self.att_mlp(torch.cat([k, q, torch.nn.ReLU()(edge_attr)], dim=-1))
        alpha = softmax(alpha, index, ptr, size_i)

        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        attn_output = alpha * v

        attn_output = attn_output.view(attn_output.shape[0], H * E)
        attn_output = self.out_proj(attn_output)

        return attn_output


# End code from https://github.com/snap-stanford/prodigy (used to implement Prodigy)

# Prodigy adapted to use different models for the data graph generation
class Prodigy(torch.nn.Module):
    def __init__(self, expert, layerS, n_layer, dropout, heads, self_loops, msg_pos_only, batch_norm_metagraph, device,
                 mu=0, sigma=0):
        """
        self: itself.
        expert (torch.nn.Module): The model used for creating the task graph.
        layerS (int): dimension of the hidden dimensions of the GNN of the task graphs.
        n_layers (int): number of layers for the GNN of the task graph.
        dropout (float): dropout of each layer.
        heads (int): number of heads for the attention for the task graph.
        self_loops,msg_pos_only,batch_norm_metagrap (boolean): determines if the respective characteristic is used in the GNN of the task graph.
        device (torch.device): device used for the calculations, e.g., cpu or cuda.
        mu (float): mean for the gaussian used to initialize the label nodes.
        sigma (float): stdv for the gaussian used to initialize the label nodes .

        Return: Prodigy(torch.nn.Module).
        """

        super().__init__()
        self.device = device
        self.layerS = layerS
        self.dropout = dropout
        self.expert = expert
        i = 0
        for p in self.expert.parameters():
            self.register_parameter(name="part" + str(i), param=p)
            i += 1

        self.emb_dim = layerS
        edge_attr_dim = 2
        self.linear = Linear(3 * layerS, layerS, bias=True, )  # for edges

        self.meta2 = MetaGNN(emb_dim=self.emb_dim, edge_attr_dim=edge_attr_dim, n_layers=n_layer, heads=heads,
                             dropout=dropout,
                             msg_pos_only=msg_pos_only, self_loops=False,
                             batch_norm=batch_norm_metagraph)  # meta2 is for the task graph
        self.cos = torch.nn.CosineSimilarity(dim=0)  # used for the similarity between label nodes and queries
        self.seed = 42
        self.training = False
        np.random.seed(seed)

        self.mu, self.sigma = mu, sigma

    # forward method of Prodigy
    def forward(self, x, edge_index, maskP, maskQ, queriesList, prompts, drop=0, mask=0, task="node"):
        """
        self: itself.
        x (torch.tensor): The features of the nodes.
        edge_index (torch.tensor): The edges.
        maskP (torch.tensor): The mask/batch for the prompts/examples.
        maskQ (torche.tensor): The mask/batch for the queries.
        prompts (dict): A dictionary containing each class's list of prompts.
        queriesList (list): A list of the query sets for the classifcation.

        drop (float): It is between 0 and 1 and is the probability that a vertex is dropped.
        mask (float): It is between 0 and 1 and is the probability that the features of a vertex  are masked by setting them to 0.
        task (String): Determines the current task: "node" (node classification), "link" or "linkS" (link prediction), "graph" (graph classification), "reason" (reasoning).

        Return: The prediction for the queries.
        """
        num_classes = len(prompts)
        ways = len(prompts[0])
        device = self.device
        np.random.seed(seed)

        # The feature initialization of the label nodes
        labelN = np.random.normal(self.mu, self.sigma,
                                  size=(num_classes, self.layerS))  # for initiliazing the class nodes in PRODIGY
        self.bn = torch.nn.BatchNorm1d(num_classes).to(self.device)
        sNumber = 0
        lenQ = 0
        self.meta2.training = self.training
        for i in queriesList:
            lenQ += i.size(dim=0)
            # simDs = None
        # if task == "graph" or task == "link" or  task == "linkS":
        #    simDs = torch.zeros(lenQ)
        # else:
        simDs = torch.zeros(lenQ, num_classes)

        # masking and droping
        if mask < 0 or mask > 1:
            mask = 0
        if drop < 0 or drop > 1:
            drop = 0
        random.seed(42)
        useE = []
        if task != "graph":
            for i in range(len(edge_index[0])):
                if random.uniform(0, 1) > drop:
                    useE.append(i)
            maskV = []
            for i in range(len(x)):
                if random.uniform(0, 1) < mask:
                    maskV.append(i)
            if len(maskV) > 0:
                x = x.detach().clone()
                x[maskV] = torch.zeros(len(x[0])).to(self.device)
        else:
            useE = [[], []]
            for i in range(len(edge_index[0][0])):
                if random.uniform(0, 1) > drop:
                    useE[0].append(i)
            for i in range(len(edge_index[1][0])):
                if random.uniform(0, 1) > drop:
                    useE[1].append(i)

            for i in range(2):
                maskV = []
                for i in range(len(x)):
                    if random.uniform(0, 1) < mask:
                        maskV.append(i)

                if len(maskV) > 0:
                    x[i] = x.detach().clone()
                    x[i][maskV] = torch.zeros(len(x[i][0])).to(self.device)

        # Getting the features for the data graph depending on the task

        self.expert.out = False
        xQ = None
        xP = None
        loss = 0
        if task == "node":
            xP, loss = self.expert(x, edge_index[:, useE], maskP, task)
            xP = xP[maskP]
            xQ, loss = self.expert(x, edge_index[:, useE], maskQ, task)
            xQ = xQ[maskQ]
        elif task == "graph":
            xP, loss = self.expert(x[0], edge_index[0][:, useE[0]], maskP, task)
            xP = global_mean_pool(xP, maskP)
            xQ, loss = self.expert(x[1], edge_index[1][:, useE[1]], maskQ, task)
            xQ = global_mean_pool(xQ, maskQ)
        elif task == "link" or task == "linkS" or task == "reason":
            xP, loss = self.expert(x, edge_index[:, useE], maskP, task)
            xQ, loss = self.expert(x, edge_index[:, useE], maskQ, task)
        self.expert.out = True
        xEMaxP = None
        if task == "link" or task == "linkS" or task == "reason":
            xEMaxP = torch.max(xP, 0)[0]  # max pooling
            xEMaxQ = torch.max(xQ, 0)[0]

        # Start of the inferrence process for each query set in queriesList
        for queries in queriesList:

            self.meta2.training = self.training
            xd = torch.zeros(num_classes * (ways + 1) + len(queries), self.emb_dim)
            edge_indexd = torch.zeros(2, 2 * ways * num_classes * num_classes + len(queries) * num_classes)
            edge_feats = torch.zeros(2 * ways * num_classes * num_classes + len(queries) * num_classes, 2)

            # Label of the edges in the task graph: [0,0] = q?, [1,1] = pT, [1,0] =pF, [0,1] self loop
            k = 0
            device = self.device

            # Prompt subgraphs
            for j in range(num_classes):
                xd[j * (ways + 1)] = torch.from_numpy(labelN[j])
                for i in range(ways):
                    p = prompts[j][i]

                    if task == "node" or task == "graph":
                        xd[j * (ways + 1) + i + 1, :] = xP[p]
                    elif task == "link" or task == "linkS" or task == "reason":
                        xd[j * (ways + 1) + i + 1] = self.linear(
                            torch.cat((xP[p[0]], xP[p[1]], xEMaxP), 0))  # Prodigy original
                        # xd[j*(ways+1)+i+1] = (xP[p[0]]- xP[p[1]])#/ xEMax #my maybe for links ?
                    edge_indexd[0][k] = j * (ways + 1)
                    edge_indexd[1][k] = j * (ways + 1) + i + 1
                    edge_feats[k][0] = 1
                    edge_feats[k][1] = 1
                    k += 1
                    edge_indexd[0][k] = j * (ways + 1) + i + 1
                    edge_indexd[1][k] = j * (ways + 1)
                    edge_feats[k][0] = 1
                    edge_feats[k][1] = 1
                    k += 1
                    for c in range(num_classes):
                        if c != j:
                            edge_indexd[0][k] = c * (ways + 1)
                            edge_indexd[1][k] = j * (ways + 1) + i + 1
                            edge_feats[k][0] = 1
                            k += 1
                            edge_indexd[0][k] = j * (ways + 1) + i + 1
                            edge_indexd[1][k] = c * (ways + 1)
                            edge_feats[k][0] = 1
                            k += 1

            # Query subgraphs
            j = 0
            for q in queries:
                if task == "node" or task == "graph":
                    xd[-(j + 1)] = xQ[int(q)]
                elif task == "link" or task == "linkS" or task == "reason":
                    xd[-(j + 1)] = self.linear(
                        torch.cat((torch.cat((xQ[int(q[0])], xQ[int(q[1])]), 0), xEMaxQ), 0))  # Prodigy original
                    # xd[-(j+1)] = (xQ[int(q[0])] - xQ[int(q[1])])#/ xEMax #my idea
                for c in range(num_classes):
                    qp = len(xd) - (j + 1)
                    edge_indexd[0][k] = c * (ways + 1)
                    edge_indexd[1][k] = qp
                    k += 1
                j += 1
            edge_indexd = edge_indexd.long()

            # inference in the data graph using meta2
            xd = self.meta2(x=xd.to(device), edge_index=edge_indexd.to(device),
                            edge_attr=edge_feats.to(device))  # meta2 is not fixed
            simD = torch.zeros(len(queries), num_classes)

            # Calculating the similarity between label nodes and queries
            for j in range(len(queries)):
                for i in range(num_classes):
                    simD[j][i] = self.cos(xd[i * (ways + 1)].double(), xd[-(j + 1)].double())
            simD = F.softmax(simD, dim=1)
            for l in range(len(simD)):
                # if task == "graph" or task == "link" or  task == "linkS":
                #    #print(simD[l][1])
                #    simDs[sNumber] = simD[l]
                # else:
                simDs[sNumber] = simD[l]
                sNumber += 1
            torch.cuda.empty_cache()

        torch.cuda.empty_cache()
        # if task == "graph" or task == "link" or  task == "linkS":
        return simDs.to(self.device), loss
        # else:
        # return self.bn(simDs.to(self.device)),loss

    # (un-)freeze the parameters of the model
    def freeze(self, freeze=False):
        """
        freeze (boolean): Determines if gradient descent is possible for the parameters of the model

        Return: None
        """

        for p in self.parameters():
            p.requires_grad = not freeze



