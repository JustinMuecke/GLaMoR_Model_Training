import numpy as np
import pandas as pd
import os
import torch
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score
import wandb
from datetime import datetime

# Load embeddings
embeddings = {}
path = "../../GLaMoR/data/filtered_embeddings/"

for file in os.listdir(path):
    if file.endswith(".npy"):
        array = np.load(os.path.join(path, file))
        embeddings[file] = array

print(len(embeddings))
# Load the CSV and align features
dataset_df = pd.read_csv("../../GLaMoR/data/filtered_dataset.csv", header=0)
x = [
    embeddings.get(filename.split(".")[0] + ".npy", np.full((100,), np.NaN))
    for filename in dataset_df["file_name"].values.tolist()
]
data = pd.DataFrame(x, columns=[f"feature_{i}" for i in range(100)])
data = data.dropna(subset=[f"feature_{i}" for i in range(100)]).reset_index(drop=True)

# Get labels and match them with remaining rows
labels = dataset_df['consistency'].loc[data.index].values  # align labels with remaining rows

# Convert to NumPy
X = data.values
y = labels

print(X.shape)
print(y.shape)
# Split into train, val, and test
models_dict = {
    "SVC": {"kernel": "rbf", "C": 1, "random_state": None},
    "LogisticRegression": {"C": 0.03359, "penalty": "l2", "max_iter": 50, "solver": "lbfgs", "random_state": None},
    "RandomForestClassifier": {"max_depth": 9, "max_features": None, "max_leaf_nodes": 20, "n_estimators": 100, "random_state": None},
    "GaussianNB": {"var_smoothing": 1e-9},
    "DecisionTreeClassifier": {"max_depth": 5, "min_samples_leaf": 5, "min_samples_split": 20, "criterion": "entropy", "random_state": None}
}

for model_name, hyperparameters in models_dict.items():
    for i in range(5):
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5)
        # Create a new instance of the model with the specific hyperparameters
        model_class = globals()[model_name]  # Get the class by name
        model = model_class(**hyperparameters)
        wandb.init(project="Baselines", name=type(model).__name__)


        train_start = datetime.now()
        model.fit(X_train, y_train)
        train_end = datetime.now()
        inf_start = datetime.now()
        y_pred = model.predict(X_test)
        inf_end = datetime.now()

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)

        wandb.log({
            "test/accuracy": accuracy, "test/precision": precision, "test/recall": recall,
            "time/training": str(train_end-train_start), "time/inference": str(inf_end - inf_start)
        })
        wandb.finish()
