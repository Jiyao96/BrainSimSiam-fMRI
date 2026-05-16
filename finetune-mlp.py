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

train_loader=DataLoader(train_dataset,batch_size=1,shuffle=False)
test_loader=DataLoader(test_dataset,batch_size=1,shuffle=False)

# define models and optimizers
subject_model = SimSiam(input_dim=268, hidden_dim=args.dim, output_dim=args.dim, embed_dim=args.dim)
subject_model.load_state_dict(torch.load("saved_models/"+args.path, map_location=device))
subject_model.to(device)

def embed():
    subject_model.eval()
    # list to store embedding
    train_list=[]
    train_label=[]
    test_list=[]
    test_label=[]
    
    count=0
    print("Generate trained embedding")
    for base_data, _ in train_loader:
        base_data = base_data.to(device)
        x1 = base_data
        z1, _, _ = subject_model(x1, x1)
        train_list.append(z1.detach())
        train_label.append(x1.y.detach())
        count+=1
        if count%500==0:
            print(f"{count*100//len(train_dataset)}%")
    # print(f"acc: {correct/len(train_loader)}")
    count=0
    print("Generate testing embedding")
    for base_data, _ in test_loader:
        base_data = base_data.to(device)
        x1 = base_data
        z1, _, _ = subject_model(x1, x1)
        test_list.append(z1.detach())
        test_label.append(x1.y.detach())
        count+=1
        if count%500==0:
            print(f"{count*100//len(test_dataset)}%")

    train_embed=torch.stack(train_list).cpu().squeeze()
    train_y=torch.stack(train_label).cpu()
    print(train_embed.shape)
    print(train_y.shape)

    test_embed=torch.stack(test_list).cpu().squeeze()
    test_y=torch.stack(test_label).cpu()
    print(test_embed.shape)
    print(test_y.shape)
    
    return train_embed, train_y, test_embed, test_y

train_embed, train_y, test_embed, test_y = embed()

def pca_and_tSNE(data):
    # Step 1: Reduce dimensionality from 2048 to 100 using PCA
    pca = PCA(n_components=100)
    data_reduced_pca = pca.fit_transform(data)
    print(f"Shape after PCA: {data_reduced_pca.shape}")
    # Step 2: Further reduce dimensionality from 100 to 2 using t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    data_reduced_tsne = tsne.fit_transform(data_reduced_pca)
    print(f"Shape after t-SNE: {data_reduced_tsne.shape}")
    return data_reduced_pca, data_reduced_tsne

################## Visualization ##################

train_pca, train_tsne=pca_and_tSNE(train_embed)
test_pca, test_tsne=pca_and_tSNE(test_embed)

correlations = np.corrcoef(train_pca.T, train_y[:,:-1].T)[:100, 100:]
print(f"Correlation matrix shape: {correlations.shape}")
# Step 1: Visualize the correlation in a heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(correlations, annot=False, cmap='coolwarm')
plt.title('Correlation between PCA Components and Label Dimensions')
plt.xlabel('Label Dimensions')
plt.ylabel('PCA Components')
plt.savefig('visualization/embed_train_'+args.path+'.png')

# Visualize the 2D data with labels
plt.figure(figsize=(10, 8))
scatter = plt.scatter(train_tsne[:, 0], train_tsne[:, 1], c=train_y[:,0], cmap='viridis', label=train_y[:,0], alpha=0.5)
plt.title('2D Visualization using t-SNE')
plt.xlabel('t-SNE Dimension 1')
plt.ylabel('t-SNE Dimension 2')
plt.colorbar(scatter, label='Label')
plt.savefig('visualization/tsne_train_'+args.path+'.png')

correlations = np.corrcoef(test_pca.T, test_y[:,:-1].T)[:100, 100:]
print(f"Correlation matrix shape: {correlations.shape}")
# Step 2: Visualize the correlation in a heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(correlations, annot=False, cmap='coolwarm')
plt.title('Correlation between PCA Components and Label Dimensions')
plt.xlabel('Label Dimensions')
plt.ylabel('PCA Components')
plt.savefig('visualization/embed_test_'+args.path+'.png')

# Visualize the 2D data with labels
plt.figure(figsize=(10, 8))
scatter = plt.scatter(test_tsne[:, 0], test_tsne[:, 1], c=test_y[:,0], cmap='viridis', label=test_y[:,0], alpha=0.5)
plt.title('2D Visualization using t-SNE')
plt.xlabel('t-SNE Dimension 1')
plt.ylabel('t-SNE Dimension 2')
plt.colorbar(scatter, label='Label')
plt.savefig('visualization/tsne_test_'+args.path+'.png')

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

# Define the custom dataset
class EmbeddingDataset(Dataset):
    def __init__(self, embeddings, labels, type='long'):
        self.embeddings = embeddings
        self.labels = labels
        self.label_dtype=type
        
    def __len__(self):
        return len(self.embeddings)
    
    def __getitem__(self, idx):
        embedding = self.embeddings[idx]
        label = self.labels[idx]
        if self.label_dtype=='long':
            return embedding.float(), label.long()
        else:
            return embedding.float(), label.float()

# Hyperparameters
input_dim = args.dim*2
hidden_dim = 2048
output_dim = 2 # Assuming binary classification for gender (e.g., 0 for male, 1 for female)
learning_rate = 0.0001
num_epochs = 10

# A. gender classification
# Create dataset and dataloader
train_dataset = EmbeddingDataset(train_embed, train_y[:,0])
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
# Create dataset and dataloader
test_dataset = EmbeddingDataset(test_embed, test_y[:,0])
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
# Initialize the model, criterion, and optimizer
model = PredictionMLP(input_dim, hidden_dim, output_dim)
criterion = nn.CrossEntropyLoss()
#criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.0001)
scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

# Training loop (pseudo-code)
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    for inputs, labels in train_loader:
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
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
    for inputs, labels in test_loader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        pred_list.append(predicted)
        output_list.append(outputs.data[0][1])
        label_list.append(labels)
pred_all=torch.stack(pred_list)
out_all=torch.stack(output_list)
label_all=torch.stack(label_list)
m1=f1_score(label_all, pred_all, zero_division=1.0)
fpr, tpr, thresholds = roc_curve(label_all, out_all.detach().numpy(), pos_label=1)
m2=auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, color='blue', lw=2, label='ROC curve (AUC = %0.2f)' % m2)
plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc="lower right")
plt.savefig('saved_models/'+args.path+'_auc.png')

print(f"Accuracy of the model on the test set: {100 * correct / total:.2f}, f1:{m1}, auc:{m2}")

torch.save(model.state_dict(),"mlp1")

# B. total int prediction
num_epochs=40
# Create dataset and dataloader
train_dataset = EmbeddingDataset(train_embed, train_y[:,12], type='float')
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
# Create dataset and dataloader
test_dataset = EmbeddingDataset(test_embed, test_y[:,12], type='float')
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
# Initialize the model, criterion, and optimizer
model = PredictionMLP(input_dim, hidden_dim, 1, use_sigmoid=True)
# criterion = nn.CrossEntropyLoss()
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate*20, weight_decay=0.0001)
scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

# Training loop (pseudo-code)
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs.squeeze(), labels)
        
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
    for inputs, labels in test_loader:
        outputs = model(inputs)
        total += labels.size(0)
        output_list.append(outputs.data[0][0])
        label_list.append(labels[0])
y_pred=torch.stack(output_list)
y_true=torch.stack(label_list)
mae=mean_absolute_error(y_true, y_pred)
mape=mean_absolute_percentage_error(y_true, y_pred)
correlation_matrix = np.corrcoef(y_true, y_pred)
corr = correlation_matrix[0, 1]
r_2 = r2_score(y_true, y_pred)
    
print(f"MAE: {mae}, MAPE: {mape}, Corr:{corr}, r2:{r_2}")
    
    
torch.save(model.state_dict(),"mlp2")

