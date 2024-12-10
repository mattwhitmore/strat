import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

class RetDataset(Dataset):
    def __init__(self,fns,transform=None,subsets = ['Ret','Ret1D','Ret5D','Ret20D','Ret60D','Ret250D','Ret500D','Ret1000D']):
        
        dfs = []
        for fn in fns:
            dfi = pd.read_csv(fn)
            dfi = dfi[subsets]
            dfs.append(dfi)

        df = pd.concat(dfs)
        df.reset_index(drop=True,inplace=True)
        self.data = df
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        #pandas insists on the stupid index col
        features = self.data.iloc[idx, 1:].values.astype(np.float32)
        target = self.data.iloc[idx, 0:1].values.astype(np.float32)
        
        sample = {'features': features, 'target': target}
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample