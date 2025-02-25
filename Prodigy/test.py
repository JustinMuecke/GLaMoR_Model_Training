import site
import sys
import importlib
from TAGLAS import get_task,get_evaluator
import Prodigy
import warnings
warnings.filterwarnings('ignore')
from TAGLAS.tasks.text_encoder import SentenceEncoder
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from typing import List, Tuple
from data_loading import load_data

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("cuda")
else:
    device = torch.device("cpu")
    print("CPU")
print(torch.__version__)
torch.manual_seed(42)



def pad_tensors(train_x, eval_x, test_x, batch_size=1000) -> Tuple[List[np.ndarray], int]:
    x = train_x + eval_x + test_x
    max_length = max(len(entry) for entry in x)

    print("Padding batches")
    padded = []
    for i in range(0, len(x), batch_size):
        batch = x[i:i + batch_size]  # Process a batch at a time
        padded_batch = [np.pad(entry, ((0, max_length - len(entry)), (0, 0)), mode='constant') for entry in batch]
        padded.extend(padded_batch)  # Store processed batch

    # Convert each NumPy array to a PyTorch tensor
    tensor_data = [[torch.from_numpy(arr) for arr in sublist] for sublist in padded]

# Stack into a single tensor (if all shapes match)
    return tensor_data, max_length

def get_tensor(train_x, eval_x, test_x):
    import torch

    x = train_x + eval_x + test_x
    max_length = max(len(entry) for entry in x)  # Determine max sequence length
    feature_dim = x[0].shape[1]  # Assuming each entry is (seq_len, feature_dim)

    # Preallocate a contiguous 3D tensor
    padded = torch.zeros((len(x), max_length, feature_dim), dtype=torch.float16)

    # Efficiently fill the tensor
    for i, entry in enumerate(x):
        seq_len = len(entry)
        padded[i, :seq_len, :] = torch.tensor(entry)  # Copy only the existing values
    
    return padded, max_length

def node():
    print("Loading Train Data")
    train_x, train_y = load_data("train")
    print("Loading Eval Data")
    eval_x, eval_y = load_data("eval")
    print("Loading Test Data")
    test_x, test_y = load_data("test")

    print("Concatinationg X Tensors")
    x, max_length = get_tensor(train_x, eval_x, test_x)


    in_channelsN = max_length
    hidden_channelsN = 512#1024
    out_channelsN = 2
    dropout = 0.5
    print(x)
    train_mask = [1 for i in range(len(train_y))] + [0 for i in range(len(eval_y))] + [0 for i in range(len(test_y))]
    eval_mask = [0 for i in range(len(train_y))] + [1 for i in range(len(eval_y))] + [0 for i in range(len(test_y))]
    test_mask = [0 for i in range(len(train_y))] + [0 for i in range(len(eval_y))] + [1 for i in range(len(test_y))]


    #GNN for creating the task graph
    gnn = Prodigy.GNN(in_channels = in_channelsN ,hidden_channels=hidden_channelsN,out_channels=out_channelsN
                            ,n_layers=1,device=device,model = "GCN",dropout=dropout, out = True)

    
    #Prodigy
    ## hyperparam
    ways = 5
    num_classes = 2
    
    prompts =  Prodigy.getPrompts(train_y ,ways, num_classes)
    num_feat = in_channelsN
    layerS = in_channelsN #256#256# Size of the embeddings
    n_layer= 1#for Attention  2
    heads =1#8#for Attention 8
    dropout = 0.01#.1#0.1#0.1#default
    msg_pos_only= False#False #default
    self_loops=False
    batch_norm_metagraph=False# False default
    expert = gnn.to(device)
    modelP = Prodigy.Prodigy(expert, layerS, n_layer, dropout, heads,self_loops,msg_pos_only,batch_norm_metagraph,device)
    modelP.to(device)
    
    #Optimizer
    lr =  0.002#0.0002
    weight_decay=0#0.002# 0.02#0.02#0.02#0.02#0.02 
    
    
    optimizer = torch.optim.Adam(modelP.parameters(), lr=lr, weight_decay=weight_decay)

    #training, validation and testing
    Prodigy.train(model=modelP,
                  maxQ=1000,
                  x=x,
                  yAll=train_y,
                  edge_index=[],
                  prompt_mask=train_mask,
                  query_mask=train_mask,
                  prompts=prompts,
                  numberQ=3,
                  drop=0,
                  mask=0,
                  epochs=10,
                  optimizer=optimizer,
                  task="node",
                  val_x=x,
                  val_y=eval_y,
                  val_mask=eval_mask,
                  val_edge=[],
                  patience=float('inf'),
                  name="Prodigy-node")

    Prodigy.evaluate(model=modelP,
                     x=x,
                     y=eval_y,
                     edge_index=[],
                     prompt_mask=train_mask,
                     query_mask=eval_mask,
                     prompts=prompts,
                     numberQ=1,
                     task="node")

    Prodigy.test(model=modelP,
                 x=x,
                 y=test_y,
                 edge_index=[],
                 prompt_mask=train_mask,
                 query_mask=test_mask,
                 prompts=prompts,
                 numberQ=1,
                 task="node")


#[1...., 0.....]
#[0,....1, ...0]
#[0,..........1]

def main():
    node()

if __name__ == "__main__":
    main()
# %%
