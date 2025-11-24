import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import os
from polaris.paths import FINETUNE_SPLIT_DIR

'''
In this code:
PTB-XL has four subset: superclass, subclass, form, rhythm
ICBEB is CPSC2018 dataset mentioned in the original paper
Chapman is the CSN dataset from the original paper
'''

class ECGDataset(Dataset):
    def __init__(self, 
                 split: str,
                 dataset_name: str = 'ptbxl-super', 
                 data_pct: float = 1,
                 ):
        """
        Args:
            data_path (string): Path to store raw data.
            csv_file (string): Path to the .csv file with labels and data path.
            mode (string): ptbxl/icbeb/chapman.
            data_pct (float): Percentage of data to use.
        """
        #  self.data_path = data_path
        self.dataset_name = dataset_name
        self.split = split
        csv_path = FINETUNE_SPLIT_DIR / f"{dataset_name}/{split}.csv"
        csv_file = pd.read_csv(csv_path, low_memory=False)
        csv_file = csv_file.sample(frac=data_pct, random_state=42).reset_index(drop=True)
        
        self.labels_name = list(csv_file.columns[4:])
        self.num_classes = len(self.labels_name)

        self.ecg_path = csv_file["data_file"]
        self.labels = csv_file.iloc[:, 4:].values

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, idx):
        ecg_path = os.path.join(FINETUNE_SPLIT_DIR / f"{self.dataset_name}/{self.split}", self.ecg_path.iloc[idx])
        # 预处理后的 .npy 存的是包含 ecg/label 等字段的字典，需允许 pickle 并取出数组
        sample_dict = np.load(ecg_path, allow_pickle=True).item()
        ecg = torch.from_numpy(sample_dict["data"]).float()
        target = torch.from_numpy(sample_dict["label"]).float()

            
 
        return {
            "ecg": ecg,
            "label": target
        }


if __name__ == "__main__":
    from polaris.paths import FINETUNE_SPLIT_DIR
    dataset = ECGDataset(
                         split="val",
                         dataset_name='ptbxl-super',
                         )
    print(len(dataset))
    sample = dataset[0]
    print(sample['ecg'].size())
    print(sample['label'].size())