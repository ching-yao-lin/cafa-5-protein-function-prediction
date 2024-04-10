# CAFA Protein Function Prediction

## Introduction
This repository contains code and documentation related to the CAFA-5 protein function prediction project. The project aims to improve the baseline solution using various feature extraction methods and as an inmprovement futher use the protein embeddings to improve the performace of the model.

## Contributors
- **Harshavardhana A S**: Implemented BLAST-KNN, 3MER, and InterPro feature extraction methods as a preliminary stage to enhance the baseline solution.
- **Unnath**: Assisted in conducting a literature survey on alternative methods to NetGo and explored various protein embedding techniques.
- **Chris**: Developed the Exploratory Data Analysis (EDA) for CAFA 5 protein data and explored and implemented part of the Kaggle fourth-place winner's solution, along with literature documents and surveys for protein encoding techniques.

## Project Structure

  CAFA_5_3MER_Implementation.ipynb - Algorithm of 3MER function

  CAFA_5_starting_point.ipynb - EDA Exploration of CAFA 5 challenge

  CAFA_KNN_BLAST.ipynb - Algorithm implementation of BLAST KNN method to extract homologous proteins

  ankh-embeddings-make.ipynb - Make the ANKH embeddings

  CAFA5_Train_and_infer_f02.ipynb - Code for training and testing the model (WIP)

  train-test-generate-abstract-embeddings.ipynb - Code to get and encode literature data for each protein sequence


## Data Sources:

https://zenodo.org/records/10951704- Contains the derived data from Kaggle: Contains associated GO terms and the protein sequences

https://zenodo.org/records/10951709 - Contains the dataset from Kaggle: IA text, Train sequence, GO file, Test Superset etc.

https://www.kaggle.com/datasets/zmcxjt/cafa5-train-test-data - Contains a part of the encoded sequences used in PROTGOAT



