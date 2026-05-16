import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch_geometric.loader import DataLoader
from torch.optim import lr_scheduler
import torch.optim as optim
from imports.DualrepData import GraphAugDataset
from models.encoder_models import SimSiam_3dAUG, edge_dropout, node_feature_mask
import argparse
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score, auc, roc_curve, roc_auc_score, mean_absolute_error, mean_absolute_percentage_error, r2_score
import matplotlib.pyplot as plt

from models.gnn_explainer_modified import GNNExplainer

################## Generate Trained Embedding ##################

torch.manual_seed(1234)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Create a parser object
parser = argparse.ArgumentParser(description="Argument parser.")

# Add arguments
parser.add_argument('--test_id', type=int, default=5, help="Data parcellation used for testing")
parser.add_argument('--task_id', type=int, default=6, help="Task used for testing")
parser.add_argument('--dim', type=int, default=1024, help="Number of dim")
parser.add_argument('--path', type=str, default="subject_model", help="path to model weights")

# Parse the arguments
args = parser.parse_args()

# subject parcel for cross-validation
test_sub=torch.load('datasets/sample_hcp_test')

# task parcel
task_list = ['emotion','gambling','language','motor','relational','social','WM']

test_dataset = GraphAugDataset('DIR TO DATA', test_sub, task_list)
print(len(test_dataset))

test_loader=DataLoader(test_dataset,batch_size=len(test_dataset),shuffle=False)

# define models and optimizers
subject_model = SimSiam_3dAUG(input_dim=268, hidden_dim=args.dim, output_dim=args.dim, embed_dim=args.dim)
subject_model.load_state_dict(torch.load("saved_models/"+args.path, map_location=device))
subject_model.to(device)

# create explainer object
explainer = GNNExplainer(subject_model, epochs=200, lr=1e-2, feat_mask_type="feature", edge_mask_type="individual", allow_edge_mask=True, return_type= 'regression')

def explain():
    subject_model.eval()
    for base_data, _ in test_loader:
        base_data = base_data.to(device)
        # n_mask, e_mask=explainer.explain_graph(base_data)
        # print(n_mask.shape)
        # print(e_mask.shape)

    # torch.save(n_mask,f"explainer_logs/node_mask_{args.path}")
    # torch.save(e_mask,f"explainer_logs/edge_mask_{args.path}")
    torch.save(base_data.edge_index,f"explainer_logs/edge_index_{args.path}")
    torch.save(base_data.batch,f"explainer_logs/batch_{args.path}")
    return 

explain()
