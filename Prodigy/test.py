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
import itertools
from tqdm import tqdm
import wandb
from datetime import datetime

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
    x = train_x + eval_x + test_x
    tensor = torch.tensor(np.stack(x)) 
    return tensor


def node(lr, ways, weight_decay, ):
    print("Loading Train Data")
    train_x, train_y = load_data("train")
    print("Loading Eval Data")
    eval_x, eval_y = load_data("eval")
    print("Loading Test Data")
    test_x, test_y = load_data("test")

    print("Concatinationg X Tensors")
    x  = get_tensor(train_x, eval_x, test_x)
    train_y = torch.tensor(train_y).to(device)
    eval_y = torch.tensor(eval_y).to(device)
    test_y = torch.tensor(test_y).to(device)

    in_channelsN = 100
    hidden_channelsN = 512#1024
    out_channelsN = 2
    dropout = 0.5
    train_mask = [1 for i in range(len(train_y))] + [0 for i in range(len(eval_y))] + [0 for i in range(len(test_y))]
    eval_mask = [0 for i in range(len(train_y))] + [1 for i in range(len(eval_y))] + [0 for i in range(len(test_y))]
    test_mask = [0 for i in range(len(train_y))] + [0 for i in range(len(eval_y))] + [1 for i in range(len(test_y))]


    #GNN for creating the task graph
    gnn = Prodigy.GNN(in_channels = in_channelsN ,hidden_channels=hidden_channelsN,out_channels=out_channelsN
                            ,n_layers=1,device=device,model = "GCN",dropout=dropout, out = True)

    
    #Prodigy
    ## hyperparam
    
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
    
    
    
    optimizer = torch.optim.Adam(modelP.parameters(), lr=lr, weight_decay=weight_decay)
    train_start = datetime.now()
    #training, validation and testing
    Prodigy.train(model=modelP,
                  maxQ=1000,
                  x=x,
                  yAll=train_y,
                  edge_index=torch.empty((2,0), dtype=torch.float16),
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
                  val_edge=torch.empty((2,0), dtype=torch.float16),
                  patience=float('inf'),
                  name="Prodigy-node")

    Prodigy.evaluate(model=modelP,
                     x=x,
                     y=eval_y,
                     edge_index=torch.empty((2,0), dtype=torch.float16),
                     prompt_mask=train_mask,
                     query_mask=eval_mask,
                     prompts=prompts,
                     numberQ=1,
                     task="node")
    train_end = datetime.now()
    inf_start = datetime.now()
    accuracy, precision, recall, predictions = Prodigy.test(model=modelP,
                 x=x,
                 y=test_y,
                 edge_index=torch.empty((2,0), dtype=torch.float16),
                 prompt_mask=train_mask,
                 query_mask=test_mask,
                 prompts=prompts,
                 numberQ=1,
                 task="node")
    inf_end = datetime.now()
    wandb.log(
        {"test/accuracy": accuracy, "test/precision": precision, "test/recall":recall,
        "time/training":str(train_end - train_start), "time/inference": str(inf_end - inf_start)}
    )
 
    return accuracy, precision, recall, modelP, predictions

#[1...., 0.....]
#[0,....1, ...0]
#[0,..........1]

def main():

    ways = 45
    weight_decay= 0.0005
    lr =  0.0001
    run_id = 0 

    for i in range(5):
        wandb_run = wandb.init(
            project="Prodigy",
            name=f"Prodigy_{i}",
            config={
                "learning_rate": lr,
                "weight_decay": weight_decay,
                "ways": ways,
            }
        )
        accuracy, precision, recall, model, predictions = node(lr=lr ,ways=ways, weight_decay=weight_decay)
        wandb_run.finish()
        run_id += 1



#    combinations = itertools.product(lr, ways, weight_decay)


#    best_recall = 0
#    best_precision = 0
#    best_accuracy = 0
#    best_ways = 0
#    best_weight_decay = 0
#    best_lr = 0
#    best_model = None
#    best_predictions = None
#    for combination in tqdm(combinations):
#        lr, ways, weight_decay = combination
#        accuracy, precision, recall, model, predictions = node(lr=lr ,ways=ways, weight_decay=weight_decay)
#        if accuracy > best_accuracy:
#            best_recall = recall
#            best_accuracy = accuracy
#            best_precision = precision
#            best_ways = ways
#            best_lr = lr
#            best_weight_decay = weight_decay
#            best_model = model
#            best_predictions = predictions
#    print(f"Best Recall: {best_recall}")
#    print(f"Best Precision: {best_precision}")
#    print(f"Best Accuracy: {best_accuracy}")
#    print(f"Parameters: LR {best_lr}, Weight Decay {best_weight_decay}, Ways {best_ways}")
#    torch.save(best_model.state_dict(), "models/" + f"prodigy_ways-{best_ways}_lr-{best_lr}_wd-{best_weight_decay}")
#    torch.save(torch.tensor(best_predictions), "models/" + f"predictions_ways-{best_ways}_lr-{best_lr}_wd-{best_weight_decay}.pt")

if __name__ == "__main__":
    main()
# %%
