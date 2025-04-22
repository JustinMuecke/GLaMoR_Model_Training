# GLaMoR_Model_Training
Contains the Repositories created to train models on OWL Ontologies consistency checking. It is part of the [GLaMoR](github.com/JustinMuecke/GLaMoR) Project. The data creation is provided in Data Pipeline Submodule.

## Structure
```
├──GLaMoR-Model_Training
│  ├──Data_Format_Translation/
│  ├──GLaMoR-HermiT/
│  ├──GLaMoR-ModernBert/
│  ├──GraphLanguageModels/
│  ├──Llama/
│  ├──Logistic_Regression/
│  ├──OWL2Vec-Star/
│  ├──Prodigy/
│  ├──WideMLP
```

## Requirements
To train the models, a `requirements.txt` file is available for most of them. If it's missing, make sure you have the following installed:

- scikit-learn  
- pandas  
- numpy  
- torch

## Exection
For each model, a python file is provided to run the experiment. Simply navigate to the correct directory and run:
```
> python3 [model_exection_file].py