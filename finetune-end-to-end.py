import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch_geometric.loader import DataLoader
from torch.optim import lr_scheduler
import torch.optim as optim
from imports.DualrepData import GraphAugDataset
from models.encoder_models import SimSiam, edge_dropout, node_feature_mask
import argparse
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score, auc, roc_curve, roc_auc_score, mean_absolute_error, mean_absolute_percentage_error, r2_score
import matplotlib.pyplot as plt

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
sample_id=args.test_id
print("test parcel: ",sample_id)
sample_offset=sample_id%5 + 1
# parameters
train_sub=torch.load('datasets/sample_hcp')
test_sub=torch.load('datasets/sample_hcp_test')
# print subject size
print('Training sub size: ', len(train_sub))
print('Testing sub size: ', len(test_sub))

# task parcel
task_list = ['emotion','gambling','language','motor','relational','social','WM']

train_dataset = GraphAugDataset('DIR TO DATA', train_sub, task_list)
print(len(train_dataset))
test_dataset = GraphAugDataset('DIR TO DATA', test_sub, task_list)
print(len(test_dataset))

train_loader=DataLoader(train_dataset,batch_size=64,shuffle=False)
test_loader=DataLoader(test_dataset,batch_size=1,shuffle=False)

################## Downstream ##################
# Define the MLP model
class PredictionMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, use_sigmoid=False):
        super(PredictionMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(0.25)
        self.softmax = nn.Softmax(dim=1)
        self.use_sigmoid=use_sigmoid
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        if self.use_sigmoid:
            x = x
        else:
            x = self.softmax(x)
        return x
    
# Define the combined model for fewshot transfer learning
class FewshotModel(nn.Module):
    def __init__(self, encoder, predictor):
        super(FewshotModel, self).__init__()
        self.encoder = encoder
        self.predictor = predictor
        
    def forward(self, x):
        x,_,_ = self.encoder(x, x)
        x = self.predictor(x)
        return x


# Hyperparameters
input_dim = args.dim*2
hidden_dim = 2048
output_dim = 2 # Assuming binary classification for gender (e.g., 0 for male, 1 for female)
learning_rate = 0.0001
num_epochs = 10

# define models and optimizers
encoder_model = SimSiam(input_dim=268, hidden_dim=args.dim, output_dim=args.dim, embed_dim=args.dim)
encoder_model.load_state_dict(torch.load("saved_models/"+args.path, map_location=device))
encoder_model = encoder_model.to(device)
# Initialize the model, criterion, and optimizer
predictor_model = PredictionMLP(input_dim, hidden_dim, output_dim)
predictor_model = predictor_model.to(device)
# combined model
model = FewshotModel(encoder_model, predictor_model)

# A. gender classification
# Create dataset and dataloader
criterion = nn.CrossEntropyLoss()
#criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.0001)
scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

# A. asd classification
# Training loop (pseudo-code)
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    # print(f"acc: {correct/len(train_loader)}")
    for base_data, _ in train_loader:
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        base_data.to(device)
        labels = base_data.y.reshape(-1,33)[:,0]
        # Forward pass
        outputs = model(base_data)
        loss = criterion(outputs.float(), labels.long())
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        correct += (predicted == labels).sum().item()
    scheduler.step()
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}, acc: {correct/len(train_dataset)}")

# Model evaluation
model.eval()
correct = 0
total = 0
pred_list=[]
output_list=[]
label_list=[]
with torch.no_grad():
    for base_data, _ in test_loader:
        # Forward pass
        base_data.to(device)
        labels = base_data.y.reshape(-1,33)[:,0]
        outputs = model(base_data)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        pred_list.append(predicted)
        output_list.append(outputs.data[0][1])
        label_list.append(labels)
pred_all=torch.stack(pred_list).detach().cpu()
out_all=torch.stack(output_list).detach().cpu()
label_all=torch.stack(label_list).detach().cpu()

m1=f1_score(label_all, pred_all, zero_division=1.0)
fpr, tpr, thresholds = roc_curve(label_all, out_all.detach().numpy(), pos_label=1)
m2=auc(fpr, tpr)

print(f"Accuracy of the model on the test set: {100 * correct / total:.2f}, f1:{m1}, auc:{m2}")

# B. total int prediction
num_epochs=20
# Create dataset and dataloader
# define models and optimizers
encoder_model = SimSiam(input_dim=268, hidden_dim=args.dim, output_dim=args.dim, embed_dim=args.dim)
encoder_model.load_state_dict(torch.load("saved_models/"+args.path, map_location=device))
encoder_model = encoder_model.to(device)
# Initialize the model, criterion, and optimizer
predictor_model = PredictionMLP(input_dim, hidden_dim, 1, use_sigmoid=True)
predictor_model = predictor_model.to(device)
# combined model
model = FewshotModel(encoder_model, predictor_model)

criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate*20, weight_decay=0.0001)
scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

# Training loop (pseudo-code)
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for base_data, _ in train_loader:
        # Zero the parameter gradients
        optimizer.zero_grad()

        base_data.to(device)
        labels = base_data.y.reshape(-1,33)[:, 12]
        
        # Forward pass
        outputs = model(base_data)
        loss = criterion(outputs.squeeze().float(), labels.float())
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    scheduler.step()
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")

# Model evaluation
model.eval()
total = 0
output_list=[]
label_list=[]
with torch.no_grad():
    for base_data, _ in test_loader:
        base_data.to(device)
        # Forward pass
        labels = base_data.y.reshape(-1,33)[:,12]
        outputs = model(base_data)

        total += labels.size(0)
        output_list.append(outputs.data[0][0])
        label_list.append(labels[0])
y_pred=torch.stack(output_list).detach().cpu()
y_true=torch.stack(label_list).detach().cpu()
    
mae=mean_absolute_error(y_true, y_pred)
mape=mean_absolute_percentage_error(y_true, y_pred)
correlation_matrix = np.corrcoef(y_true, y_pred)
corr = correlation_matrix[0, 1]
r_2 = r2_score(y_true, y_pred)
    
print(f"MAE: {mae}, MAPE: {mape}, Corr:{corr}, r2:{r_2}")
    
    
