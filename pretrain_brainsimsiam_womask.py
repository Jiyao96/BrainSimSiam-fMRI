import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch_geometric.loader import DataLoader
from torch.optim import lr_scheduler
from imports.DualrepData import GraphAugDataset
from models.encoder_models import SimSiam, SimCLR, edge_dropout, node_feature_mask
import argparse

torch.manual_seed(123)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Create a parser object
parser = argparse.ArgumentParser(description="Argument parser.")

# Add arguments
parser.add_argument('--test_id', type=int, default=5, help="Data parcellation used for testing")
parser.add_argument('--task_id', type=int, default=6, help="Task used for testing")
parser.add_argument('--num_epoch', type=int, default=100, help="Number of epochs")
parser.add_argument('--dim', type=int, default=1024, help="Number of epochs")
parser.add_argument('--node_prob', type=float, default=0.2, help="node masking probability")
parser.add_argument('--edge_prob', type=float, default=0.2, help="edge dropping probability")
parser.add_argument('--alpha', type=float, default=1.0, help="scalar weight for loss terms")

# Parse the arguments
args = parser.parse_args()

# parameters
batch_size=64
learning_rate=1e-5
weight_decay=0.001
opt_method="SGD"
num_epoch=args.num_epoch

# parameters
train_sub=torch.load('datasets/sample_hcp')

# task parcel
task_list = ['emotion','gambling','language','motor','relational','social','WM']
train_dataset = GraphAugDataset('DIR TO DATA', train_sub, task_list)
train_loader=DataLoader(train_dataset,batch_size=batch_size,shuffle=True)

# define models and optimizers
subject_model = SimSiam(input_dim=268, hidden_dim=args.dim, output_dim=args.dim, embed_dim=args.dim).to(device)

# define two optimizers, one for each model
if opt_method == 'Adam':
    subject_opt = torch.optim.Adam(subject_model.parameters(), lr= learning_rate, weight_decay=weight_decay)
elif opt_method == 'SGD':
    subject_opt = torch.optim.SGD(subject_model.parameters(),lr=learning_rate,momentum=0.9, weight_decay=weight_decay, nesterov=True)

# define schedulars
subejct_scheduler = lr_scheduler.StepLR(subject_opt, step_size=20, gamma=0.5)

def aug(data, edge_prob=args.edge_prob, node_prob=args.node_prob):
    data.x = node_feature_mask(data.x, node_prob)
    data.edge_index, data.edge_attr = edge_dropout(data.edge_index, data.edge_attr, edge_prob)
    return data

def train():
    subject_model.train()
    # log loss metrics
    Lall_1=0
    Lall_2=0
    Lall_3=0
    for base_data, task_aug_data in train_loader:
        # send data to GPU 
        base_data = base_data.to(device)
        task_aug_data = task_aug_data.to(device)

        temp_batch=base_data.x.shape[0]//268
        ### train subject model ###
        # remove existing optimizer gradient
        subject_opt.zero_grad()
        # 1. robustness to augmentation on subject model
        x1 = aug(base_data)
        x2 = aug(base_data)
        # simsiam network
        _, _, L_1 = subject_model(x1, x2)
        # log loss
        Lall_1 += L_1.sum()
        L_1.mean().backward()
        # update gradient
        subject_opt.step()

        # remove existing optimizer gradient
        subject_opt.zero_grad()
        # 2. same task sub representation
        x1 = aug(base_data)
        x2 = aug(task_aug_data)
        _, _, L_2 = subject_model(x1, x2)
        # log loss
        Lall_2 += L_2.sum()
        L_2=args.alpha*L_2
        L_2.mean().backward()
        # update gradient
        subject_opt.step()

    # update scheduler
    subejct_scheduler.step()
    return Lall_1/len(train_dataset), Lall_2/len(train_dataset)

for i in range(1,num_epoch+1):
    print("Epoch: ", i)
    l1, l2 = train()
    print(f"Training losses L1:{l1} L2:{l2}")

torch.save(subject_model.state_dict(), f"weights_womask")
    
    
    
