import torch
import numpy as np
from torch.utils.data import Dataset
from torch_geometric.data import DataLoader, Batch
import pandas as pd
import os
from pathlib import Path

# create dataset class 
class GraphImageDataset(Dataset):
    def __init__(self, root, sub_list, task_list):
        self.root = root
        self.sub = sub_list
        self.task = task_list
        self.path = []
        for sub in sub_list:
            for task in task_list:
                self.path.append(root+sub+'_'+task+'_0')
        self.path=np.array(self.path)
        
    def __len__(self):
        return len(self.path)

    def __getitem__(self, idx):
        file_path=self.path[idx]
        return torch.load(file_path),torch.load(file_path)

def random_with_exclude(excluded, lower_bound=0, upper_bound=100):
    while True:
        random_int = torch.randint(lower_bound, upper_bound, (1,)).item()
        if random_int != excluded:
            return random_int

# create dataset class 
class GraphAugDataset(Dataset):
    def __init__(self, root, sub_list, task_list, dataset='HCP'):
        self.root = root
        self.sub = sub_list
        self.task = task_list
        self.path = [[] for _ in range(len(task_list))]
        if dataset=='biopoint':
            for sub in sub_list:
                for i in range(len(task_list)):
                    self.path[i].append(root+sub+'_'+task_list[i])
        if dataset=='CBT':
            for sub in sub_list:
                for i in range(len(task_list)):
                    self.path[i].append(root+sub+'_0_'+task_list[i])
        else:
            for sub in sub_list:
                for i in range(len(task_list)):
                    self.path[i].append(root+sub+'_'+task_list[i]+'_0')
        self.path=np.array(self.path)

        print('initialized length: ', len(self.sub))
        
    def __len__(self):
        return len(self.sub)*len(self.task)

    def __getitem__(self, idx):
        sub_idx = idx // len(self.task)
        task_idx = idx % len(self.task)
        # print("sub idx: ",sub_idx)
        # print("task idx: ",task_idx)
        # base data instance
        base_data = torch.load(self.path[task_idx][sub_idx])
        # variation in subject
        # sub_aug = random_with_exclude(sub_idx, upper_bound=len(self.sub))
        # sub_data = torch.load(self.path[task_idx][sub_aug])
        # variation in task
        if len(self.task)!=1:
            task_aug = random_with_exclude(task_idx, upper_bound=len(self.task))
        else:
            task_aug = task_idx
        task_data = torch.load(self.path[task_aug][sub_idx])
        # img data format
        base_data.img=torch.tensor(base_data.img)[None,:,:,:]
        # sub_data.img=torch.tensor(sub_data.img)[None,:,:,:]
        task_data.img=torch.tensor(task_data.img)[None,:,:,:]
        return base_data, task_data
        
        # print(f"original: {sub_idx}, {task_idx} new: {sub_aug}, {task_aug}")
        # list of task data for contrastive loss
        # task_list=[]
        # for i in range(len(self.task)):
        #     if i==task_idx:
        #         continue
        #     else:
        #         task_list.append(torch.load(self.path[i][sub_idx]))
        # return base_data, sub_data, task_data, task_list


sessions = ["REST1_LR","REST1_RL","REST2_LR","REST2_RL"]
rest_root = 'DIR TO DATA'
# create dataset class 
class GraphAugDataset_rest(Dataset):
    def __init__(self, root, sub_list, task_list, dataset='HCP'):
        self.root = root
        self.sub = sub_list
        self.task = task_list
        self.path = []
        # task data
        for sub in sub_list:
            for i in range(len(task_list)):
                self.path.append(root+sub+'_'+task_list[i]+'_0')
        # rest data
        for sub in sub_list:
            for ses in sessions:
                p = rest_root+sub+"_"+ses
                if Path(p).is_file():
                    self.path.append(p)
        
        self.path=np.array(self.path)
        print('initialized length: ', len(self.path))
        
    def __len__(self):
        return len(self.path)

    def __getitem__(self, idx):
        file_path=self.path[idx]
        data = torch.load(file_path)
        if len(data.img.shape)==3:
            data.img=torch.tensor(data.img)[None,:,:,:]
        return data

