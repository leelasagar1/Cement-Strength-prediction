# Cement Strength Prediction Project

## Description

This project aims to build a regression model to predict the concrete compressive strength based on the different features. The model uses the data to learn the relationship between the input features and the cement strength, and is then able to make predictions on new, unseen data.

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
