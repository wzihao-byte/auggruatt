
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from train import *
from test import test_model
from premodel import *
import argparse
from tools.plot_training_history import plot_training_history
from dual_domain_model import MODEL_VARIANTS

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="This script processes a dataset.")
    
    # --dataset 인자를 추가합니다.
    parser.add_argument('--dataset', type=str, required=True, help="Name of the dataset")
    parser.add_argument('--batchsize', type=int, required=True, help="Batchsize")
    parser.add_argument('--learningrate', type=float, required=True, help="1e-3")
    parser.add_argument('--epochs', type=int, required=True, help="100")
    parser.add_argument('--model_variant', type=str, default='baseline', choices=MODEL_VARIANTS, help="baseline|dual_concat|dual_gated")
    parser.add_argument('--freq_feature_dim', type=int, default=64, help="Frequency branch hidden/feature dimension")
    parser.add_argument('--fusion_hidden_dim', type=int, default=64, help="Fusion MLP hidden dimension")
    parser.add_argument('--freq_use_abs', type=str, default='false', help="Use abs() before temporal summary (set true for raw magnitude/complex-like inputs)")
    parser.add_argument('--verbose', action='store_true', help="Increase output verbosity")
    args = parser.parse_args()

    dataset = args.dataset

    #hyperparameter
    batchsize = args.batchsize
    learningrate = args.learningrate
    num_epochs = args.epochs
    model_variant = args.model_variant
    freq_feature_dim = args.freq_feature_dim
    fusion_hidden_dim = args.fusion_hidden_dim
    freq_use_abs = str(args.freq_use_abs).strip().lower() in {"1", "true", "t", "yes", "y", "on"}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    #Which Dataset??
    if dataset.lower() == 'aril':

        train_loader, test_loader = arilsetting(batchsize)        
        inputsize, classes =52, 6

    elif dataset.lower() == 'har-1':
        train_loader, test_loader = harsetting1(batchsize)
        inputsize, classes = 104, 4

    elif dataset.lower() == 'har-3':
        train_loader, test_loader = harsetting3(batchsize)
        inputsize, classes = 256, 5

    elif dataset.lower() == 'signfi':
        #only for home , you can change lab or home. 
        train_loader, test_loader = signfisetting(batchsize)
        inputsize, classes =90, 276

    elif dataset.lower() == 'stanfi':
        train_loader, test_loader = stanfisetting(batchsize)
        inputsize, classes =90, 6

    else:
        raise ValueError(f"Unknown dataset: {dataset}. Expected one of: aril, har-1, har-3, signfi, stanfi")

    # Create a deterministic validation split from the training set so checkpoint
    # selection never depends on test data.
    train_dataset = train_loader.dataset
    val_loader = None
    if len(train_dataset) >= 2:
        val_size = max(1, int(0.1 * len(train_dataset)))
        if val_size >= len(train_dataset):
            val_size = len(train_dataset) - 1
        train_size = len(train_dataset) - val_size
        generator = torch.Generator().manual_seed(42)
        train_subset, val_subset = random_split(train_dataset, [train_size, val_size], generator=generator)
        train_loader = DataLoader(train_subset, batch_size=batchsize, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=batchsize, shuffle=False)
        if args.verbose:
            print(f"Train/val split: train={len(train_subset)} val={len(val_subset)}")
    elif args.verbose:
        print("Dataset too small for validation split; training without validation checkpoint selection.")

    #Model setting, training and testing Model
    model,criterion,optimizer,scheduler = set_train_model(
        device=device,
        input_size=inputsize,
        hidden_size=128,
        attention_dim=32,
        num_classes=classes,
        learningrate=learningrate,
        epochs=num_epochs,
        model_variant=model_variant,
        freq_feature_dim=freq_feature_dim,
        fusion_hidden_dim=fusion_hidden_dim,
        freq_use_abs=freq_use_abs,
    )
    bestmodel, loss_hist, acc_hist = train_model(
        device,
        model,
        criterion,
        optimizer,
        scheduler,
        num_epochs,
        train_loader,
        valloader=val_loader,
        testloader=test_loader,
        p=[0.3, 0.7],
    )

    # Keep plotting helper backward-compatible with legacy train/test labels.
    plot_loss_hist = loss_hist
    plot_acc_hist = acc_hist
    if "val" in loss_hist:
        plot_loss_hist = {"train": loss_hist["train"], "test": loss_hist["val"]}
        plot_acc_hist = {"train": acc_hist["train"], "test": acc_hist["val"]}
    elif "test" not in loss_hist and "test_final" in loss_hist:
        # No per-epoch eval split available; draw final test score as a flat reference line.
        final_test_loss = loss_hist["test_final"][0]
        final_test_acc = acc_hist["test_final"][0]
        plot_loss_hist = {"train": loss_hist["train"], "test": [final_test_loss] * len(loss_hist["train"])}
        plot_acc_hist = {"train": acc_hist["train"], "test": [final_test_acc] * len(acc_hist["train"])}
    
    plot_training_history(plot_loss_hist, plot_acc_hist)
    test_model(bestmodel,device,test_loader)
