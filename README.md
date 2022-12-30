## Cement Strength Prediction project

This project aims to develop a machine learning model to predict the strength of cement based on various factors such as `Cement, Blast Furnace Slag, curing time,Water,Superplasticizer,Coarse Aggregate,Fine Aggregate, and Age`

The model is trained on a dataset of cement strength measurements, along with their corresponding input features. The model uses this data to learn the relationship between the input features and the cement strength, and is then able to make predictions on new, unseen data.

### Setup

To use this model clone the repository to your local machine and use following commands

#### CMD

`pip install -r requirements`

`python src/run.py --action train or predict`

#### Docker

`docker build -t cement-strength-prediction .`

`docker run -dp 3000:3000 cement-strength-prediction`

