# Cement Strength Prediction Project

## Description

This project aims to build a regression model to predict the concrete compressive strength based on the different features. The model uses the data to learn the relationship between the input features and the cement strength, and is then able to make predictions.

## Model Training steps

1. Clustering - KMeans algorithm is used to find clusters in the data. The idea behind clustering is to implement different algorithms to train data in different clusters. The Kmeans model is trained over preprocessed data and the model is saved for further use in prediction.
2. Model Selection - After clusters are created, we find the best model for each cluster. We are using three algorithms, "Random forest Regressor" and “Linear Regression” and "XGBoost Regressor".
 

## Installation

To use the Cement Strength Prediction project, follow these steps:

1. Clone the GitHub repository to your local machine.
2. Ensure that you have Python 3.x installed.
3. Set up a virtual environment (optional but recommended) and activate it.
4. Install the required Python packages by running the following command:

```shell
pip install -r requirements.txt
```

5. Run the script either using CMD or Docker:
##### CMD
 ```shell
 python src/run.py --action train or predict
 ```
##### Docker
```shell
docker build -t cement-strength-prediction .

docker run -dp 3000:3000 cement-strength-prediction
```
  




\
