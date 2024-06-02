# AMP-BERT-BIOCHEM

Welcome! 

This repository contains research material on LLMs based on the Transformer architecture, with application in antimicrobial peptides classification.
To be precise, it contains source code for the creation of custom, Transformer-based architectures that feature model-parallelization of HuggingFace
models.

It is part of a Bachelor of Science final project, and it is in active use by a research team: that is why it is still in active development.
Its structure is separated in functional folders. That is, every folder contains the necessary files for a certain role. Here is how to travel through
the repository, based on what you want to use it for:


  - To use it as a library or code references: go for the /src folder. It contains two main files:
    
      - MultiGPUModels.py: this file contains the definition of several classes. They represent components or full architectures for Large Language Models
      - pipeline_tools.py: this file contains the definition of different tools to use in a machine learning pipeline, for training and testing models
      
  - To revise the experiments of the project: go for the /notebooks folder. It contains many notebooks, each one being a step of the research. The relevant ones are:
  
      - ReproductionModel.ipynb: this is the first notebook. It shows the re-engineering of AMP-BERT, along the creation of a DL pipeline and a comparison of models
      - ResultsVisualization.ipynb: this is the second notebook. It shows how UMAP plots were used to diagnose the possible flaws of AMP-BERT
      - BioChem.ipynb: this is the third notebook. It features the design of AMP-BERT, an enhancement over our reproduction of AMP-BERT, and its comparison with it
      - FeatureAblation.ipynb: this is the fourth notebook. In it, several experiments are conducted, where the main model takes only subsets of predictors in order to analyze their importance.
      
      The rest of the notebooks represent work in progress.
