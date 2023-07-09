# Cement Strength Prediction Project

This project focuses on predicting the strength of cement based on various features. The dataset includes the following features: Cement, Blast Furnace Slag, curing time, Water, Superplasticizer, Coarse Aggregate, Fine Aggregate, and Age. The goal is to build a predictive model that can accurately estimate the strength of cement based on these input variables.

## Description

The Cement Strength Prediction project aims to develop a machine learning model that can predict the strength of cement based on specific features. The model uses the data to learn the relationship between the input features and the cement strength, and is then able to make predictions on new, unseen data.

## Installation

To use the Cement Strength Prediction project, follow these steps:

1. Clone the GitHub repository to your local machine.
2. Ensure that you have Python 3.x installed.
3. Set up a virtual environment (optional but recommended) and activate it.
4. Install the required Python packages by running the following command:

```shell
pip install -r requirements.txt
```

5. Run the script using the following command:
### CMD
 ```shell
 python src/run.py --action train or predict
 ```
### Docker
```shell
docker build -t cement-strength-prediction .

docker run -dp 3000:3000 cement-strength-prediction
```
  




\
