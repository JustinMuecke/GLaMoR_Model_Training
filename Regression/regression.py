import os 
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from typing import Dict, Tuple
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score
import pickle
from tqdm import tqdm

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
            "best_recall": clf.cv_results_["mean_test_recall"][clf.best_index_]
        }

        print(f"Best Estimator: {clf.best_params_}")
        print(f"Best Accuracy: {best_models[model]['best_accuracy']:.4f}")
        print(f"Best Precision: {best_models[model]['best_precision']:.4f}")
        print(f"Best Recall: {best_models[model]['best_recall']:.4f}")

        best_model_name = max(best_models, key=lambda name: best_models[name]["best_accuracy"])  # Or use precision/recall for different focus
        return best_models[best_model_name]["estimator"], best_model_name

def create_dataset(name : str, embeddings):
    dataset = pd.read_csv(f"data/{name}", header = 0)
    x = [embeddings.get(filename.split(".")[0]+".npy", np.full((100,), np.NaN)) for filename in dataset["file_name"].values.tolist()]
    # Convert list of arrays into a DataFrame (each row corresponds to one array)
    data = pd.DataFrame(x, columns=[f"feature_{i}" for i in range(100)])  # Adjust number of features as needed
    # Add the "y" column from the dataset
    data["y"] = dataset["consistency"].map({"Consistent" : 0, "Inconsistent" : 1})

    # Filter out rows where any feature vector is NaN
    data = data.dropna(subset=[f"feature_{i}" for i in range(100)])

    return data
if __name__ == "__main__":
 

    print("Load Embeddings")
    embeddings = {}
    for file in tqdm(os.listdir("data/embeddings")):
        if(".npy" in file):
            array = np.load("data/embeddings/"+file)
            embeddings[file] = array

    print("Load Training Dataset")
    train = create_dataset("train_data.csv", embeddings)
    print("Load Eval Dataset")
    eval = create_dataset("eval_data.csv", embeddings)

    train_data = pd.concat([train,eval])
    


    parameter_grid : Dict[str, Dict] = {
                        "logistic_regression": {
                            "model": LogisticRegression(), 
                            "parameters": {
                                'penalty':['l1','l2'], 
                                'C' : np.logspace(-4,4,20), 
                                'solver': ['lbfgs', 'liblinear'],
                                'max_iter'  : [50, 100,1000,2500]
                            }
                        }
                    }
    print("Start CV")
    model, _ =_perform_grid_search(parameter_grid, train_data.drop("y", axis=1), train_data["y"])
    with open("liner_regressor.pkl", "wb") as f:
        pickle.dump(model, f)