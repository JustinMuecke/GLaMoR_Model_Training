
from typing import List, Tuple
import os
import numpy as np
import pandas as pd

def load_data(split : str)-> Tuple[List[np.ndarray], List[int]]:

    embeddings = {}
    for file in os.listdir("data/embeddings_full"):
        array = np.load("data/embeddings_full/"+file)
        embeddings[file] = array

    df = pd.read_csv(f"data/{split}_data.csv", header=0)
    df["consistency"] = df["consistency"].map({"Consistent": 0, "Inconsistent": 1})
    df["embedding"] = [embeddings.get(filename.split(".")[0]+".npy", np.nan)for filename in df["file_name"].values.tolist()]
    df = df.dropna(subset=["embedding"])

    return (df["embedding"].values.tolist(), df["consistency"].values.tolist())

if __name__ == "__main__":
    data, label = load_data("test")
    print(data)
    print(label)