from training_model import Train_model
from prediction_model import Prediction
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--action',choices=['train','predict'], type=str,required=True)
args = parser.parse_args()

if args.action.lower() in ['train','training'] :

    train_model = Train_model()
    train_model.train()
elif args.action.lower() in ['predict','prediction']:
    prediction_model = Prediction()
    prediction_model.prediction_from_model()
else:
    print("wrong argumnets")


