import numpy as np
import pandas as pd
import torch

def test_model(model, test_loader, device):
    model.eval()
    predictions = []
    targets = []
    
    with torch.no_grad():
        for X_b, y_b in test_loader:
            X_b, y_b = X_b.to(device), y_b.to(device)

            pred = model(X_b)
            predictions.append(pred.cpu())
            targets.append(y_b.cpu())

    predictions = torch.cat(predictions, dim=0)
    targets = torch.cat(targets, dim=0)

    return predictions.numpy(), targets.numpy()

def eval_model(preds:np.array, targets:np.array)->dict:
    results = {}
    
    results["MAE"] = np.mean(np.abs(preds - targets))
    results["MSE"] = np.mean((preds - targets)**2)
    results["RMSE"] = np.sqrt(results["MSE"])
    
    actual_dir = np.sign(np.diff(targets, axis=0))
    pred_dir = np.sign(np.diff(preds, axis=0))
    results["DIR"] = np.mean(actual_dir == pred_dir)
    
    return results

def benchmark(models, test_loader):
    results = {}

    for model in models:
        pred, target = test_model(model, test_loader)
        results[model.__class__.__name__] = eval_model(pred, target)
    return pd.DataFrame(results).T