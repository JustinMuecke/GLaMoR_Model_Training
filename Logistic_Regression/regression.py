import os 
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from typing import Dict, Tuple
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score
import pickle
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import time
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.tree import DecisionTreeClassifier

def _perform_grid_search(parameters : Dict[str, Dict], x_train :pd.DataFrame, y_train : pd.DataFrame, scoring : str = "accuracy") -> Tuple[object, str]:    
    '''
    Given the parameters containing the ranges, performs a gridsearch
    
    Args: 
     - parameters
     - x_train
    '''
    scoring = {
    "accuracy": make_scorer(accuracy_score),
    "precision": make_scorer(precision_score, average='macro'),  # or 'micro', 'weighted', depending on your needs
    "recall": make_scorer(recall_score, average='macro')  # same as precision
    }

    best_models = {}

    for model, config in parameters.items():
        print(f"Computing Model: {model}")
        start = time.time()
        clf = GridSearchCV(estimator=config["model"], 
                        param_grid=config["parameters"], 
                        scoring=scoring,  # Pass scorers here
                        cv=3,
                        n_jobs=-1,
                        refit="accuracy")  # refit on the best accuracy score
        
        clf.fit(x_train, y_train)
        best_models[model] = {
            "estimator": clf.best_estimator_,
            "score": clf.best_score_,
            "best_params": clf.best_params_,
            "best_accuracy": clf.cv_results_["mean_test_accuracy"][clf.best_index_],
            "best_precision": clf.cv_results_["mean_test_precision"][clf.best_index_],
            "best_recall": clf.cv_results_["mean_test_recall"][clf.best_index_],
            "train_time" : clf.cv_results_["mean_fit_time"][clf.best_index_]
        }

        print(f"Best Estimator: {clf.best_params_}")
        print(f"Best Accuracy: {best_models[model]['best_accuracy']:.4f}")
        print(f"Best Precision: {best_models[model]['best_precision']:.4f}")
        print(f"Best Recall: {best_models[model]['best_recall']:.4f}")
        print(f"Training Time: {best_models[model]['train_time']:.4f}")
   
        best_model_name = max(best_models, key=lambda name: best_models[name]["best_accuracy"])  # Or use precision/recall for different focus
        return best_models[best_model_name]["estimator"], best_model_name



if __name__ == "__main__":
 
    path = "../../GLaMoR/data/filtered_embeddings/"
    embeddings = {}
    for file in os.listdir(path):
        if(".npy" in file):
            array = np.load(path+file)
            embeddings[file] = array

    dataset = pd.read_csv("../../GLaMoR/data/filtered_dataset.csv", header = 0)
    x = [embeddings.get(filename.split(".")[0]+".npy", np.full((100,), np.NaN)) for filename in dataset["file_name"].values.tolist()]
    # Convert list of arrays into a DataFrame (each row corresponds to one array)
    train_data = pd.DataFrame(x, columns=[f"feature_{i}" for i in range(100)])  # Adjust number of features as needed

# Add the "y" column from the dataset
    train_data["y"] = dataset["consistency"]

# Filter out rows where any feature vector is NaN
    train_data = train_data.dropna(subset=[f"feature_{i}" for i in range(100)])
    train_data.drop(train_data.tail(int(len(train_data)*0.15)).index, inplace=True)
    parameter_grid : Dict[str, Dict] = {
                        #"logistic_regression": {
                        #    "model": LogisticRegression(), 
                        #    "parameters": {
                        #        'penalty':['l1','l2'], 
                        #        'C' : np.logspace(-4,4,20), 
                        #        'solver': ['lbfgs', 'liblinear'],
                        #        'max_iter'  : [50, 100,1000,2500]
                        #    }
                        #}
                       # "random_forest" : {
                       # "model": RandomForestClassifier(), 
                       # "parameters": {
                       #     'n_estimators': [25, 50, 100, 150], 
                       #     "max_features": ['sqrt', 'log2', None], 
                       #     'max_depth': [None, 3, 6, 9, 12], 
                       #     'max_leaf_nodes': [5, 10, 20]
                       #    }
                       #}
  #                     "SVC": {
  #                         "model": SVC(probability=True),
  #                         "parameters": {
  #                         "C": [0.1, 1, 10],
  #                         "kernel": ["linear", "rbf"]
  #                         }
  #                     }
#                       "gaussian_nb": {
 #                          "model": GaussianNB(),
  #                         "parameters": {
   #                            "var_smoothing": [1e-9, 1e-8, 1e-7, 1e-6, 1e-5]
    #                       }
     #                  }
                        "Decision Tree": {
                            "model": DecisionTreeClassifier(),
                            "parameters": {
                                "criterion": ["gini", "entropy"],  # Splitting criteria
                                "max_depth": [None, 5, 10, 20, 50],  # Tree depth
                                "min_samples_split": [2, 5, 10, 20],  # Minimum samples to split
                                "min_samples_leaf": [1, 2, 5, 10],  # Minimum samples in a leaf
                                "max_features": ["sqrt", "log2", None],  # Number of features to consider
                                "class_weight": [None, "balanced"],  # Handling class imbalance
                            }
                       }
                  }
    model, _ = _perform_grid_search(parameter_grid, train_data.drop("y", axis=1), train_data["y"])
    with open("decision_tree.pkl", "wb") as file:
        pickle.dump(model, file)
