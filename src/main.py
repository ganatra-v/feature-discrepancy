import argparse
import json
import os

import torch
from train import train_model

parser = argparse.ArgumentParser(description="Run feature discrepancy experiments")

parser.add_argument("--dataset", type=str, help="Dataset to be used for the experiment")
parser.add_argument("--batch_size", type=int, help="Batch size for training")
parser.add_argument("--num_features", type=int, help="Number of features in the dataset")
parser.add_argument("--hidden_layers", type=int, help="Number of hidden layers in the neural network")
parser.add_argument("--hidden_size", type=int, help="Number of neurons in each hidden layer")
parser.add_argument("--num_classes", type=int, help="Number of classes in the dataset")
parser.add_argument("--epochs", type=int, help="Number of epochs for training")
parser.add_argument("--num_experiments", type=int, help="Number of experiments to run")
parser.add_argument("--lr", type=float, help="Learning rate for training")

parser.add_argument("--output_dir", type=str, help="Output directory to save results")

if __name__ == "__main__":
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    args.output_dir = os.path.join(args.output_dir, args.dataset, f"hidden_layers_{args.hidden_layers}_hidden_size_{args.hidden_size}_batch_size_{args.batch_size}_lr_{args.lr}")
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "args.json"), "w") as f:
        json.dump(vars(args), f)
    
    feature_ranks = dict()

    for i in range(args.num_experiments):
        torch.manual_seed(i)
        result = train_model(args)
        feature_ranks[i+1] = result
    
    with open(os.path.join(args.output_dir, "feature_ranks.json"), "w") as f:
        json.dump(feature_ranks, f)
