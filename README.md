## Environment
* Python 3.6
* PyTorch ≥ 1.2.0
* NumPy
* RDKit 2019.03.3.0
* pandas
* Gensim ≥ 3.4.0

## Model Workflow
## 1. Data Preparation
* Feature generation is performed using featurizer.py.
* Run the following script to generate the feature files:
* python featurizer.py
## 2. Model Training
* Train the model using:
* python main.py
* The main workflow includes:
* Loading dataset information (e.g., Human, DrugBank, etc.)
* Five-fold cross-validation was performed, with the dataset partitioned into five subsets. In each fold, three subsets (60%) were used for training, one subset (20%) for validation, and one subset (20%) for testing.
## 3. Attention Output Module
* This project provides an independent attention-based interpretability module, attention.py, which can be used to automatically extract cross-attention weights from a trained DTI classification model, specifically capturing protein → compound attention.
* This functionality can be implemented by writing a separate script (e.g., extract_attention.py) that loads the best-performing model and calls the analysis functions in attention.py.
Key Features
* No retraining is required; attention weights can be automatically extracted by loading best.pt and importing attention.py.
* Cross-attention from protein → compound is extracted from the Transformer decoder module.
* Attention weight files are automatically saved, facilitating downstream visualization and analysis.

## Authors
This code was originally created by Lu Wang and Yuxue Pan, who were master students at Zhejiang University of Science and Technology.

Wang was under joint supervision of Dr. Qu Chen and Prof. Yifeng Zhou, and Pan was under joint supervision of Dr. Qu Chen and Prof. Juan Huang.
This code serves as the Supporting Information for the manuscript entitled "AI-Assisted Molecular Docking and Molecular Dynamics Simulations for Predicting Off-Target Effects of AKT1 ATP-Competitive Inhibitors" and can be downloaded for free.

edited on January 5th, 2026

