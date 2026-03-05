
import torch.nn as nn
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
        print(f"Unknown dataset: {dataset}")        

    #Model setting, training and testing Model
    model,criterion,optimizer,scheduler = set_train_model(
        device=device,
        input_size=inputsize,
        hidden_size=128,
        attention_dim=32,
        num_classes=classes,
        learningrate=1e-3,
        epochs=100,
        model_variant=model_variant,
        freq_feature_dim=freq_feature_dim,
        fusion_hidden_dim=fusion_hidden_dim,
        freq_use_abs=freq_use_abs,
    )
    bestmodel, loss_hist, acc_hist =train_model(device, model, criterion, optimizer, scheduler, num_epochs, train_loader, test_loader, p=[0.3, 0.7])
    
    plot_training_history(loss_hist, acc_hist)
    test_model(bestmodel,device,test_loader)
