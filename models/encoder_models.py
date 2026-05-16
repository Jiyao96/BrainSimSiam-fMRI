import torch
import torch.nn as nn
import torch.nn.functional as F 
from torch_geometric.nn import GCNConv, SAGEConv, GATConv
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
from torch_geometric.utils import add_self_loops
from monai.transforms import (
    Compose,
    RandFlip,
    RandRotate90,
    RandScaleIntensity,
    RandShiftIntensity,
    EnsureType,
)

# CGL dependency
from torch_geometric.nn import TopKPooling, ChebConv
from torch_geometric.utils import add_self_loops, sort_edge_index, remove_self_loops
from torch_sparse import spspmm

#############################
######### Utilities #########
#############################

### monai image augmentations for ablation study
transforms = Compose([
    RandFlip(spatial_axis=0, prob=0.5),
    RandFlip(spatial_axis=1, prob=0.5),
    RandFlip(spatial_axis=2, prob=0.5),
    RandRotate90(prob=0.5, spatial_axes=[0, 1]),
    RandScaleIntensity(factors=0.1, prob=0.5),
    RandShiftIntensity(offsets=0.1, prob=0.5),
    EnsureType(),
])

### loss function for SimSiam training
def D(p, z, version='simplified'): # negative cosine similarity
    if version == 'original':
        z = z.detach() # stop gradient
        p = F.normalize(p, dim=1) # l2-normalize 
        z = F.normalize(z, dim=1) # l2-normalize 
        return -(p*z).sum(dim=1).mean()
    elif version == 'simplified':# same thing, much faster. Scroll down, speed test in __main__
        return - F.cosine_similarity(p, z.detach(), dim=-1).mean()
    else:
        raise Exception

### SimCLR contrastive loss
# modified from https://towardsdatascience.com/nt-xent-normalized-temperature-scaled-cross-entropy-loss-explained-and-implemented-in-pytorch-cc081f69848
def nt_xent(rep1, rep2, labels, temperature):
    # label list to bool matrix
    label_mat = (labels.unsqueeze(1) == labels.unsqueeze(0)).float()
    # cosine similarity
    sim_mat = F.cosine_similarity(rep1[None,:,:], rep2[:,None,:], dim=-1)
    sim_mat = sim_mat/temperature
    # ce loss
    loss = F.binary_cross_entropy(sim_mat.sigmoid(), label_mat, reduction="none")
    # divide by class
    target_pos = label_mat.bool()
    target_neg = ~label_mat.bool()
    # aggregate
    loss_pos = torch.zeros(rep1.size(0), rep1.size(0)).to(rep1.device).masked_scatter(target_pos, loss[target_pos])
    loss_neg = torch.zeros(rep1.size(0), rep1.size(0)).to(rep1.device).masked_scatter(target_neg, loss[target_neg])
    loss_pos = loss_pos.sum(dim=1)
    loss_neg = loss_neg.sum(dim=1)
    num_pos = label_mat.sum(dim=1)
    num_neg = rep1.size(0) - num_pos
    # return with balanced loss
    return ((loss_pos / num_pos) + (loss_neg / num_neg)).mean()

### EDGE DROPOUT augmentation
def edge_dropout(edge_index, edge_attr, prob=0.2):
    # Dropping edges randomly
    if prob > 0:
        # Generate a mask for the edges
        edge_mask = torch.rand(edge_index.size(1)) >= prob
        edge_index = edge_index[:, edge_mask]
        edge_attr = edge_attr[edge_mask]
    return edge_index, edge_attr

### NODE FEATURE MASK augmentation
def node_feature_mask(x, prob=0.2):
    x_dim = x.shape[0]
    # dropping features on all nodes
    mask = torch.rand(x.size(1)) < prob  # Generate a random mask
    x[:, mask] = torch.zeros((x_dim,1), device=x.device)
    return x

### NODE IMG MASK augmentation
def node_img_mask(data, prob=0.2, num_roi=84, default_mask=None):
    x = data.x
    img = data.img

    # process
    x_dim = x.shape[0]
    num_graph = x_dim // num_roi
    # dropping node features in graph 
    for i in range(num_graph):
        # drop nodes as graph node feature 
        x_seg = x[i*num_roi:(i+1)*num_roi,:]
        mask = torch.rand(num_roi) < prob  # Generate a random mask
        x_seg[mask,:] = torch.zeros((1, x.shape[1]), device=x.device)
        x[i*num_roi:(i+1)*num_roi,:] = x_seg

        # drop roi as image regions
        img_seg = img[i]
        # print(img_seg.shape)
        if default_mask is not None:
            img_mask = default_mask[mask].sum(dim=0) == 0
        else:
            mask_list = data.mask[i*num_roi:(i+1)*num_roi]
            img_mask = mask_list[mask].sum(dim=0) == 0
        # print(img_mask.shape)
        # print(torch.max(mask_list[mask].sum(dim=0)))
        # print(torch.min(mask_list[mask].sum(dim=0)))
        img_seg = img_seg * img_mask 
        img[i] = img_seg
    return x, img

### GAT encoder backbone architecture
class GNN_backbone(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=1):
        super().__init__()
        self.conv1 = GATConv(in_channels, hidden_channels//4, heads=1, edge_dim=1)
        self.conv2 = GATConv(hidden_channels//4, out_channels//4, heads=1, edge_dim=1)

    def forward(self, x, edge_index, edge_weight, batch):
        # First GAT convolutional layer
        x1 = self.conv1(x=x, edge_index=edge_index, edge_attr=edge_weight)
        x1 = F.relu(x1)
        # Second GAT convolutional layer
        x2 = self.conv2(x=x1, edge_index=edge_index, edge_attr=edge_weight)
        out = torch.cat([gmp(x1, batch), gap(x1, batch), gmp(x2, batch), gap(x2, batch)], dim=1)
        return out

### GraphSAGE encoder backbone architecture
class SAGE_backbone(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels//4)
        self.conv2 = SAGEConv(hidden_channels//4, out_channels//4)

    def forward(self, x, edge_index, edge_weight, batch):
        # First GAT convolutional layer
        x1 = self.conv1(x=x, edge_index=edge_index)
        x1 = F.relu(x1)
        # Second GAT convolutional layer
        x2 = self.conv2(x=x1, edge_index=edge_index)
        x2 = F.relu(x2)
        out = torch.cat([gmp(x1, batch), gap(x1, batch), gmp(x2, batch), gap(x2, batch)], dim=1)
        return out

### GCN encoder backbone architecture
class GCN_backbone(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels//4)
        self.conv2 = GCNConv(hidden_channels//4, out_channels//4)

    def forward(self, x, edge_index, edge_weight, batch):
        # First GAT convolutional layer
        x1 = self.conv1(x=x, edge_index=edge_index, edge_weight=edge_weight)
        x1 = F.relu(x1)
        # Second GAT convolutional layer
        x2 = self.conv2(x=x1, edge_index=edge_index, edge_weight=edge_weight)
        x2 = F.relu(x2)
        out = torch.cat([gmp(x1, batch), gap(x1, batch), gmp(x2, batch), gap(x2, batch)], dim=1)
        return out

### CNN encoder backbone architecture
class CNN_backbone(nn.Module):
    def __init__(self, in_channels=1, out_channels=1024):
        super().__init__()
        # Convolutional layers
        self.conv1 = nn.Conv3d(in_channels, 8, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv3d(8, 16, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv3d(16, 48, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv3d(48, 96, kernel_size=3, stride=1, padding=1)
        
        # Pooling layer
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2, padding=0)
        
        # Fully connected layers
        self.fc1 = nn.Linear(96 * 5 * 6 * 5, 512)
        self.fc2 = nn.Linear(512, out_channels)
        
    def forward(self, x):
        x=x[:,None,:,:,:]
        # Apply convolutional layers with ReLU activation and pooling layers
        x = self.pool(F.relu(self.conv1(x)))  # Size: [batch_size, 8, 45, 54, 45]
        x = self.pool(F.relu(self.conv2(x)))  # Size: [batch_size, 16, 22, 27, 22]
        x = self.pool(F.relu(self.conv3(x)))  # Size: [batch_size, 48, 11, 13, 11]
        x = self.pool(F.relu(self.conv4(x)))  # Size: [batch_size, 96, 5, 6, 5]
        # Flatten the tensor
        x = x.view(x.size(0), -1)  # Size: [batch_size, 96 * 5 * 6 * 5]
        # Fully connected layers with ReLU activation and Dropout
        x = F.relu(self.fc1(x))  # Size: [batch_size, 512]
        x = self.fc2(x)  # Size: [batch_size, out_channels]
        return x

### MLP modules 
class projection_MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim=4096, out_dim=8192):
        super().__init__()
        ''' 
        Projection MLP. The projection MLP (in f) has BN ap-
        plied to each fully-connected (fc) layer, including its out- 
        put fc. Its output fc has no ReLU. The hidden fc is 2048-d. 
        This MLP has 3 layers.
        '''
        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.layer3 = nn.Sequential(
            nn.Linear(hidden_dim, out_dim),
            nn.BatchNorm1d(out_dim)
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x 

class prediction_MLP(nn.Module):
    def __init__(self, in_dim=8192, hidden_dim=512, out_dim=8192): # bottleneck structure
        super().__init__()
        ''' 
        Prediction MLP. The prediction MLP (h) has BN applied 
        to its hidden fc layers. Its output fc does not have BN
        (ablation in Sec. 4.4) or ReLU. This MLP has 2 layers. 
        The dimension of h’s input and output (z and p) is d = 2048, 
        and h’s hidden layer’s dimension is 512, making h a 
        bottleneck structure (ablation in supplement). 
        '''
        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Linear(hidden_dim, out_dim)
        """
        Adding BN to the output of the prediction MLP h does not work
        well (Table 3d). We find that this is not about collapsing. 
        The training is unstable and the loss oscillates.
        """

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x 


##############################
########### Models ###########
##############################

### SimSiam model
class SimSiam(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, embed_dim):
        super().__init__()
        # encoder module
        self.gnn = GNN_backbone(in_channels=input_dim, hidden_channels=hidden_dim, out_channels=output_dim)
        self.cnn = CNN_backbone(out_channels=output_dim)
        self.projector = projection_MLP(in_dim=output_dim*2, out_dim=embed_dim*2)
        # predictor module
        self.predictor = prediction_MLP(in_dim=embed_dim*2, out_dim=embed_dim*2)
    
    def forward(self, x1, x2, mask=None):
        # x1 encoder
        if mask is not None:
            gnn1 = self.gnn(x1.x*mask, x1.edge_index, x1.edge_attr, x1.batch)
        else:
            gnn1 = self.gnn(x1.x, x1.edge_index, x1.edge_attr, x1.batch)
        cnn1 = self.cnn(x1.img)
        z1 = torch.cat((gnn1,cnn1),dim=1)
        z1 = self.projector(z1)
        
        # x2 encoder
        if mask is not None:
            gnn2 = self.gnn(x2.x*mask, x2.edge_index, x2.edge_attr, x2.batch)
        else:
            gnn2 = self.gnn(x2.x, x2.edge_index, x2.edge_attr, x2.batch)
        cnn2 = self.cnn(x2.img)
        z2 = torch.cat((gnn2,cnn2),dim=1)
        z2 = self.projector(z2)
        
        # predictor
        h = self.predictor
        p1, p2 = h(z1), h(z2)
        L = D(p1, z2) / 2 + D(p2, z1) / 2
        return z1, z2, L

### SimSiam model with 3D augmentation ablation
class SimSiam_3dAUG(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, embed_dim):
        super().__init__()
        # encoder module
        self.gnn = GNN_backbone(in_channels=input_dim, hidden_channels=hidden_dim, out_channels=output_dim)
        self.cnn = CNN_backbone(out_channels=output_dim)
        self.projector = projection_MLP(in_dim=output_dim*2, out_dim=embed_dim*2)
        # predictor module
        self.predictor = prediction_MLP(in_dim=embed_dim*2, out_dim=embed_dim*2)
    
    def forward(self, x1, x2, mask=None):
        # apply 3d augmentations to CNN
        x1_img=transforms(x1.img)
        x2_img=transforms(x2.img)
        # x1_img = x1.img
        # x2_img = x2.img
        
        # x1 encoder
        if mask is not None:
            gnn1 = self.gnn(x1.x*mask, x1.edge_index, x1.edge_attr, x1.batch)
        else:
            gnn1 = self.gnn(x1.x, x1.edge_index, x1.edge_attr, x1.batch)
        cnn1 = self.cnn(x1_img)
        z1 = torch.cat((gnn1,cnn1),dim=1)
        z1 = self.projector(z1)
        
        # x2 encoder
        if mask is not None:
            gnn2 = self.gnn(x2.x*mask, x2.edge_index, x2.edge_attr, x2.batch)
        else:
            gnn2 = self.gnn(x2.x, x2.edge_index, x2.edge_attr, x2.batch)
        cnn2 = self.cnn(x2_img)
        z2 = torch.cat((gnn2,cnn2),dim=1)
        z2 = self.projector(z2)
        
        # predictor
        h = self.predictor
        p1, p2 = h(z1), h(z2)
        L = D(p1, z2) / 2 + D(p2, z1) / 2
        return z1, z2, L

### SimSiam model without projection MLP
class SimSiam_wo_proj(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, embed_dim):
        super().__init__()
        # encoder module
        self.gnn = GNN_backbone(in_channels=input_dim, hidden_channels=hidden_dim, out_channels=output_dim)
        self.cnn = CNN_backbone(out_channels=output_dim)
        # predictor module
        self.predictor = prediction_MLP(in_dim=embed_dim*2, out_dim=embed_dim*2)
    
    def forward(self, x1, x2, mask=None):
        # x1 encoder
        if mask is not None:
            gnn1 = self.gnn(x1.x*mask, x1.edge_index, x1.edge_attr, x1.batch)
        else:
            gnn1 = self.gnn(x1.x, x1.edge_index, x1.edge_attr, x1.batch)
        cnn1 = self.cnn(x1.img)
        z1 = torch.cat((gnn1,cnn1),dim=1)
        
        # x2 encoder
        if mask is not None:
            gnn2 = self.gnn(x2.x*mask, x2.edge_index, x2.edge_attr, x2.batch)
        else:
            gnn2 = self.gnn(x2.x, x2.edge_index, x2.edge_attr, x2.batch)
        cnn2 = self.cnn(x2.img)
        z2 = torch.cat((gnn2,cnn2),dim=1)
        
        # predictor
        h = self.predictor
        p1, p2 = h(z1), h(z2)
        L = D(p1, z2) / 2 + D(p2, z1) / 2
        return z1, z2, L

### SimSiam model with GNN encoder only
class SimSiam_GNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, embed_dim):
        super().__init__()
        # encoder module
        self.gnn = GNN_backbone(in_channels=input_dim, hidden_channels=hidden_dim, out_channels=output_dim)
        self.projector = projection_MLP(in_dim=output_dim, out_dim=embed_dim*2)
        # predictor module
        self.predictor = prediction_MLP(in_dim=embed_dim*2, out_dim=embed_dim*2)
    
    def forward(self, x1, x2, mask=None):
        # x1 encoder
        if mask is not None:
            gnn1 = self.gnn(x1.x*mask, x1.edge_index, x1.edge_attr, x1.batch)
        else:
            gnn1 = self.gnn(x1.x, x1.edge_index, x1.edge_attr, x1.batch)
        z1 = gnn1
        z1 = self.projector(z1)
        
        # x2 encoder
        if mask is not None:
            gnn2 = self.gnn(x2.x*mask, x2.edge_index, x2.edge_attr, x2.batch)
        else:
            gnn2 = self.gnn(x2.x, x2.edge_index, x2.edge_attr, x2.batch)
        z2 = gnn2
        z2 = self.projector(z2)
        
        # predictor
        h = self.predictor
        p1, p2 = h(z1), h(z2)
        L = D(p1, z2) / 2 + D(p2, z1) / 2
        return z1, z2, L

### SimSiam model with CNN encoder only
class SimSiam_CNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, embed_dim):
        super().__init__()
        # encoder module
        self.cnn = CNN_backbone(out_channels=output_dim)
        self.projector = projection_MLP(in_dim=output_dim, out_dim=embed_dim*2)
        # predictor module
        self.predictor = prediction_MLP(in_dim=embed_dim*2, out_dim=embed_dim*2)
    
    def forward(self, x1, x2, mask=None):
        # x1 encoder
        cnn1 = self.cnn(x1.img)
        z1 = cnn1
        z1 = self.projector(z1)
        
        # x2 encoder
        cnn2 = self.cnn(x2.img)
        z2 = cnn2
        z2 = self.projector(z2)
        
        # predictor
        h = self.predictor
        p1, p2 = h(z1), h(z2)
        L = D(p1, z2) / 2 + D(p2, z1) / 2
        return z1, z2, L

### SimCLR model
class SimCLR(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, embed_dim, temperature):
        super().__init__()
        self.temperature=temperature
        # encoder module
        self.gnn = GNN_backbone(in_channels=input_dim, hidden_channels=hidden_dim, out_channels=output_dim)
        self.cnn = CNN_backbone(out_channels=output_dim)
        self.projector = projection_MLP(in_dim=output_dim*2, out_dim=embed_dim)
        # predictor module
        self.predictor = prediction_MLP(in_dim=embed_dim*2, out_dim=embed_dim*2)
    
    def forward(self, x1, x2, mask=None):
        # x1 encoder
        if mask is not None:
            gnn1 = self.gnn(x1.x*mask, x1.edge_index, x1.edge_attr, x1.batch)
        else:
            gnn1 = self.gnn(x1.x, x1.edge_index, x1.edge_attr, x1.batch)
        cnn1 = self.cnn(x1.img)
        z1 = torch.cat((gnn1,cnn1),dim=1)
        # x2 encoder
        if mask is not None:
            gnn2 = self.gnn(x2.x*mask, x2.edge_index, x2.edge_attr, x2.batch)
        else:
            gnn2 = self.gnn(x2.x, x2.edge_index, x2.edge_attr, x2.batch)
        cnn2 = self.cnn(x2.img)
        z2 = torch.cat((gnn2,cnn2),dim=1)

        # predictor
        h = self.predictor
        p1, p2 = h(z1), h(z2)

        # task label
        y = x1.y.reshape((-1,15))[:,-1].long()
        # loss function
        L1 = D(p1, z2) / 2 + D(p2, z1) / 2
        L2 = nt_xent(p1, p2, y, self.temperature)
        
        return z1, z2, p1, p2, L1, L2

### SimSiam encoder model with MLP head for supervised training  
class SupervisedModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, embed_dim):
        super().__init__()
        # encoder module
        self.gnn = GNN_backbone(in_channels=input_dim, hidden_channels=hidden_dim, out_channels=output_dim)
        self.cnn = CNN_backbone(out_channels=output_dim)
        self.projector = projection_MLP(in_dim=output_dim*2, out_dim=embed_dim)
    
    def forward(self, x1):
        # x1 encoder
        gnn1 = self.gnn(x1.x, x1.edge_index, x1.edge_attr, x1.batch)
        cnn1 = self.cnn(x1.img)
        z1 = torch.cat((gnn1,cnn1),dim=1)
        z1 = self.projector(z1)
        z1 = F.sigmoid(z1)
        return z1
    
class SupervisedSAGE(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, embed_dim):
        super().__init__()
        # encoder module
        self.gnn = SAGE_backbone(in_channels=input_dim, hidden_channels=hidden_dim, out_channels=output_dim)
        self.cnn = CNN_backbone(out_channels=output_dim)
        self.projector = projection_MLP(in_dim=output_dim*2, out_dim=embed_dim)
    
    def forward(self, x1):
        # x1 encoder
        gnn1 = self.gnn(x1.x, x1.edge_index, x1.edge_attr, x1.batch)
        cnn1 = self.cnn(x1.img)
        z1 = torch.cat((gnn1,cnn1),dim=1)
        z1 = self.projector(z1)
        z1 = F.sigmoid(z1)
        return z1
    
class SupervisedGCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, embed_dim):
        super().__init__()
        # encoder module
        self.gnn = GCN_backbone(in_channels=input_dim, hidden_channels=hidden_dim, out_channels=output_dim)
        self.cnn = CNN_backbone(out_channels=output_dim)
        self.projector = projection_MLP(in_dim=output_dim*2, out_dim=embed_dim)
    
    def forward(self, x1):
        # x1 encoder
        gnn1 = self.gnn(x1.x, x1.edge_index, x1.edge_attr, x1.batch)
        cnn1 = self.cnn(x1.img)
        z1 = torch.cat((gnn1,cnn1),dim=1)
        z1 = self.projector(z1)
        z1 = F.sigmoid(z1)
        return z1

# CGL baseline models
# CGL baseline
def compute_contrastive_loss(out_1, out_2, temperature = 0.1):
    """
    compute the contrastive loss of two views of the same sample
    Args:
        out1: view 1, shape: (B, dim)
        out2: view 2, shapeL (B, dim)
    Returns:

    """
    out_1 = F.normalize(out_1, dim=1)
    out_2 = F.normalize(out_2, dim=1)
    # [2*B, D]
    batch_size = out_1.shape[0]
    out = torch.cat([out_1, out_2], dim=0)
    # [2*B, 2*B]
    sim_matrix = torch.exp(torch.mm(out, out.t().contiguous()) / temperature)
    # print("sim_matrix1:", torch.isnan(sim_matrix).any())
    mask = (torch.ones_like(sim_matrix) - torch.eye(2 * batch_size, device=sim_matrix.device)).bool()
    # print("mask:", torch.isnan(mask).any())
    # [2*B, 2*B-1]
    sim_matrix = sim_matrix.masked_select(mask).view(2 * batch_size, -1)
    # print("sim_matrix2:", torch.isnan(sim_matrix).any())

    # compute loss
    pos_sim = torch.exp(torch.sum(out_1 * out_2, dim=-1) / temperature)
    # print("pos_sim:", torch.isnan(pos_sim).any())
    # [2*B]
    pos_sim = torch.cat([pos_sim, pos_sim], dim=0)
    # print(pos_sim / sim_matrix.sum(dim=-1))
    loss = (- torch.log(pos_sim / sim_matrix.sum(dim=-1))).mean()
    # print("loss:", torch.isnan(loss).any())
    return loss
    
class CGL_backbone(torch.nn.Module):
    def __init__(self, indim=84, ratio=0.5, R=84, embed_dim=2048):
        '''
        :param indim: (int) node feature dimension,
        :param ratio: (float) pooling ratio in (0,1)
        :param R: (int) number of ROIs
        :param auxdim: (int) number of PCD features
        '''
        super(CGL_backbone, self).__init__()
        self.indim = indim
        self.latent_dim = embed_dim//4
        self.R = R
        self.conv1 = ChebConv(indim, self.latent_dim, K=3)
        self.pool1 = TopKPooling(self.latent_dim, ratio=ratio, multiplier=1, nonlinearity=torch.sigmoid)
        self.conv2 = ChebConv(self.latent_dim, self.latent_dim, K=3)
        self.pool2 = TopKPooling(self.latent_dim, ratio=ratio, multiplier=1, nonlinearity=torch.sigmoid)

    def augment_adj(self, edge_index, edge_weight, num_nodes):
        edge_index, edge_weight = add_self_loops(edge_index, edge_weight,num_nodes=num_nodes)
        edge_index, edge_weight = sort_edge_index(edge_index, edge_weight,num_nodes)
        edge_index, edge_weight = spspmm(edge_index, edge_weight, edge_index,edge_weight, num_nodes, num_nodes,num_nodes)
        edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)
        return edge_index, edge_weight

    def forward(self, x, edge_index, edge_attr, batch):
        x = self.conv1(x, edge_index, edge_attr)
        x, edge_index, edge_attr, batch, perm, score1 = self.pool1(x, edge_index, edge_attr, batch)
        x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        edge_attr = edge_attr.squeeze()
        edge_index, edge_attr = self.augment_adj(edge_index, edge_attr, x.size(0))

        x = self.conv2(x, edge_index, edge_attr)
        x, edge_index, edge_attr, batch, perm, score2 = self.pool2(x, edge_index,edge_attr, batch)

        x2 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
        out = torch.cat([x1,x2], dim=1)

        # Check if any element is NaN
        has_nan = torch.isnan(out).any()
        return out

class CGL_baseline(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, embed_dim):
        super().__init__()
        # encoder module
        self.gnn = CGL_backbone(indim=input_dim, ratio=0.5, R=input_dim, embed_dim=embed_dim)
        self.projector = projection_MLP(in_dim=output_dim, out_dim=embed_dim*2)
    
    def forward(self, x1, x2, mask=None):
        # x1 encoder
        gnn1 = self.gnn(x1.x, x1.edge_index, x1.edge_attr, x1.batch)
        z1 = gnn1
        z1 = self.projector(z1)
        
        # x2 encoder
        gnn2 = self.gnn(x2.x, x2.edge_index, x2.edge_attr, x2.batch)
        z2 = gnn2
        z2 = self.projector(z2)
            
        L = compute_contrastive_loss(z1, z2)
        return z1, z2, L

class CGL_CNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, embed_dim):
        super().__init__()
        # encoder module
        self.gnn = CGL_backbone(indim=input_dim, ratio=0.5, R=input_dim, embed_dim=embed_dim)
        self.cnn = CNN_backbone(out_channels=output_dim)
        self.projector = projection_MLP(in_dim=output_dim*2, out_dim=embed_dim*2)
    
    def forward(self, x1, x2, mask=None):
        # x1 encoder
        gnn1 = self.gnn(x1.x, x1.edge_index, x1.edge_attr, x1.batch)
        cnn1 = self.cnn(x1.img)
        z1 = torch.cat((gnn1,cnn1),dim=1)
        z1 = self.projector(z1)
        
        # x2 encoder
        gnn2 = self.gnn(x2.x, x2.edge_index, x2.edge_attr, x2.batch)
        cnn2 = self.cnn(x2.img)
        z2 = torch.cat((gnn2,cnn2),dim=1)
        z2 = self.projector(z2)
        
        L = compute_contrastive_loss(z1, z2)
        return z1, z2, L


