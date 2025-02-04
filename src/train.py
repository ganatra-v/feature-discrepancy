from dataloader import load_data
from model import neural_network
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, log_loss
import shap

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset

from tqdm import tqdm

def evaluate_performance(net, x_train, y_train, x_test, y_test):
    y_pred_train = net.infer(x_train)
    loss_train = log_loss(y_train, y_pred_train)
    acc_train = accuracy_score(y_train, y_pred_train > 0.5)

    y_pred_test = net.infer(x_test)
    loss_test = log_loss(y_test, y_pred_test)
    acc_test = accuracy_score(y_test, y_pred_test > 0.5)

    print(f"loss_train {loss_train}")
    print(f"acc_train {acc_train}")
    print(f"loss_test {loss_test}")
    print(f"acc_test {acc_test}")

    return acc_train, acc_test

def calculate_shap_values(model, x_train, x_test):
    shap_explainer = shap.Explainer(model.infer, x_train)
    shap_values = shap_explainer(x_test)
    shap_values = np.abs(shap_values.values).mean(axis=0)
    feature_ranks = np.flip(np.argsort(shap_values)).tolist()
    return shap_values.tolist(), feature_ranks

def train_model(args):
    # load dataset
    x_train, y_train, x_test, y_test = load_data(args)
    traindataset = TensorDataset(torch.from_numpy(x_train).float(), torch.from_numpy(y_train).float())
    trainloader = DataLoader(traindataset, batch_size = args.batch_size, shuffle = True)

    x_train_tensor = torch.from_numpy(x_train).float().cuda()
    y_train_tensor = torch.from_numpy(y_train).float().cuda()
    x_test_tensor = torch.from_numpy(x_test).float().cuda()
    y_test_tensor = torch.from_numpy(y_test).float().cuda()

    # load model
    model = neural_network(args.num_features, args.hidden_layers, args.hidden_size, args.num_classes).cuda()

    # loss function
    loss_fn = nn.BCEWithLogitsLoss()

    total_params = sum([p.numel() for p in model.parameters()])
    optimizer = torch.optim.Adam(model.parameters(), lr = 1e-4, weight_decay = 1e-3)
    for epoch in tqdm(range(args.epochs)):
        for i,batch in enumerate(trainloader):
            x,y = batch
            x = x.cuda()
            y = y.cuda()
            y_pred = model(x)
            loss = loss_fn(y_pred, y.view(-1,1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    acc_train, acc_test = evaluate_performance(model, x_train_tensor, y_train, x_test_tensor, y_test)
    shap_values, feature_ranks = calculate_shap_values(model, x_train, x_test)

    return {
        "train_accuracy" : acc_train,
        "test_accuracy" : acc_test,
        "num_features" : x_train.shape[-1],
        "feature_ranks" : feature_ranks,
        "total_params" : total_params,
        "hidden_size" : args.hidden_size,
        "num_hidden_layers" : args.hidden_layers,
        "shap_values" : shap_values
    }