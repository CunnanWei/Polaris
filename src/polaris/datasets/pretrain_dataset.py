'''
The script contains pretraining dataset for MIMIC-IV-ECG
'''
import os
import torch
import pandas as pd
from torch.utils.data import Dataset
from typing import List
import numpy as np
from tqdm import tqdm
from einops import rearrange
import wfdb
import itertools
from polaris.paths import PRETRAIN_SPLIT_DIR, DATA_ROOT_PATH
from polaris.datasets.augmentations import RandomLeadsMask


class ECG_Text_Dataset(Dataset):
    """ Dataset for MIMIC-IV-ECG"""
    def __init__(self, 
                 split: str, 
                 dataset_dir: str, 
                 dataset_list: List = ["mimic-iv-ecg"], 
                 data_pct: float = 1, 
                 transforms = None,
                 n_views: int = 1,
                 use_cmsc: bool = False,
                 use_rlm: bool = False,
                 num_beats: int = 1,
                 ):
        
        super().__init__()
        
        self.split = split
        self.dataset_dir = dataset_dir
        self.dataset_list = dataset_list
        self.data_pct = data_pct
        self.use_cmsc = use_cmsc
        self.use_rlm = use_rlm
        self.n_views = n_views
        if transforms is None:
            self.augs = []
        else:
            self.augs = transforms
        if self.use_rlm:
            # random mask 50% leads for each samplesa
            self.augs.append(
                RandomLeadsMask(p=1, mask_leads_selection="random", mask_leads_prob=0.5)
                )

        all_df = []
        for dataset_name in self.dataset_list:
            df = pd.read_csv(PRETRAIN_SPLIT_DIR / f"{self.split}.csv", low_memory=False)
            df["path"] = df["path"].apply(lambda x: os.path.join(self.dataset_dir, x))
            print(f"Loading {dataset_name} {self.split} dataset: total {len(df)} samples")
            all_df.append(df)
        self.df = pd.concat(all_df)
        # sample data
        self.df = self.df.sample(frac=self.data_pct).reset_index(drop=True)

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        patient_id = torch.tensor([row["study_id"]]).long()
        report = row["total_report"]
        ecg = np.load(row["path"])

        ecg = torch.tensor(ecg).float()
        num_leads = ecg.shape[0]

        if self.use_cmsc:
            num_samples = ecg.size(1)
            ecg1 = ecg[:, :num_samples//2]
            ecg2 = ecg[:, num_samples//2:]
            for aug in self.augs:
                ecg1 = aug(ecg1)
                ecg2 = aug(ecg2)
            ecg = torch.stack([ecg1, ecg2], dim=0)
            patient_id = torch.cat([patient_id, patient_id], dim=0)
        else:
            if self.n_views == 1:
                for aug in self.augs:
                    ecg = aug(ecg)
            else:
                ecg_list = []
                for _ in range(self.n_views):
                    # original ecg
                    ecg_ = ecg.clone()
                    for aug in self.augs:
                        ecg_ = aug(ecg_)
                    ecg_list.append(ecg_)
                ecg = torch.stack(ecg_list, dim=0)
                patient_id = torch.cat([patient_id]*self.n_views, dim=0)

        return {
            "patient_id": patient_id,
            "ecg": ecg,
            "report": report
        }



if __name__ == "__main__":
    # from polaris.datasets.augmentations import TRandomResizedCrop, TTimeOut
    # rr_crop_ratio_range = [0.5, 1.0]
    # output_size = 250*5
    # to_crop_ratio_range = [0, 0.5]
    # transforms = [
    #     TRandomResizedCrop(
    #     crop_ratio_range=rr_crop_ratio_range, 
    #     output_size=output_size),
    #     TTimeOut(crop_ratio_range=to_crop_ratio_range)
    # ]
    dataset = ECG_Text_Dataset(split="val", dataset_dir=str(DATA_ROOT_PATH),  
                               dataset_list=["mimic-iv-ecg"],
                               use_cmsc=False,
                               use_rlm=False,
                               data_pct=0.1,
                               num_beats=1
                               )
    print(len(dataset))
    sample = dataset[0]
