import os
from typing import List
from lightning.pytorch.utilities.types import EVAL_DATALOADERS
import pandas as pd
from torch.utils.data import DataLoader
from lightning import LightningDataModule
from polaris.datasets.pretrain_dataset import ECG_Text_Dataset
from polaris.datasets.finetune_dataset import ECGDataset
from polaris.paths import DATA_ROOT_PATH


class ECGTextDataModule(LightningDataModule):
    def __init__(self, 
                 dataset_dir: str, 
                 dataset_list: List = ["mimic-iv-ecg"],
                 val_dataset_list: List = ["ptbxl-super", "ptbxl-sub", "ptbxl-form", "ptbxl-rhythm",
                                           "icbeb", "chapman"],
                 batch_size: int = 128, 
                 num_workers: int = 4,
                 train_data_pct: float = 1.,
                 use_cmsc: bool = False,
                 use_rlm: bool = False,
                 transforms = None,
                 n_views: int = 1,
                 ):
        super().__init__()

        self.dataset_dir = dataset_dir
        self.dataset_list = dataset_list
        self.val_dataset_list = val_dataset_list
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_data_pct = train_data_pct
        self.use_cmsc = use_cmsc
        self.use_rlm = use_rlm
        self.transforms = transforms
        self.n_views = n_views

    def train_dataloader(self):

        train_dataset = ECG_Text_Dataset(
            split="train",
            dataset_dir=self.dataset_dir,
            dataset_list=self.dataset_list,
            data_pct=self.train_data_pct,
            use_cmsc=self.use_cmsc,
            use_rlm=self.use_rlm,
            transforms=self.transforms,
            n_views=self.n_views
        )

        train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            # collate_fn=custom_collate_fn,
            shuffle=True,
            pin_memory=True,
            num_workers=self.num_workers,
        )

        return train_dataloader

    def val_dataloader(self):
        if self.val_dataset_list is None:
            val_dataset = ECG_Text_Dataset(
                split="val",
                dataset_dir=self.dataset_dir,
                dataset_list=self.dataset_list,
                data_pct=1,
                use_cmsc=self.use_cmsc,
                use_rlm=self.use_rlm,

                transforms=self.transforms,
                n_views=self.n_views,
            )

            val_dataloader = DataLoader(
                val_dataset,
                batch_size=self.batch_size,
                # collate_fn=custom_collate_fn,
                shuffle=False,
                pin_memory=True,
                num_workers=self.num_workers,
            )

            return val_dataloader
        else:
            val_loaders = []
            for dataset_name in self.val_dataset_list:

                val_dataset = ECGDataset(
                    split="val",
                    dataset_name=dataset_name,
                    data_pct=1
                )

                val_dataloader = DataLoader(
                    val_dataset,
                    batch_size=self.batch_size,
                    shuffle=False,
                    pin_memory=True,
                    num_workers=self.num_workers,
                    drop_last=False
                )

                val_loaders.append(val_dataloader)

            return val_loaders

    def test_dataloader(self):
        test_dataset = ECG_Text_Dataset(
            split="test",
            dataset_dir=self.dataset_dir,
            dataset_list=self.dataset_list,
            data_pct=1,
            use_cmsc=self.use_cmsc,
            use_rlm=self.use_rlm,
            transforms=self.transforms,
            n_views=self.n_views
        )

        test_dataloader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            # collate_fn=custom_collate_fn,
            shuffle=False,
            pin_memory=True,
            num_workers=self.num_workers,
        )

        return test_dataloader

if __name__ == "__main__":
    dm = ECGTextDataModule(
        dataset_dir=str(DATA_ROOT_PATH),
        dataset_list=["mimic-iv-ecg"],
        val_dataset_list=None,
        batch_size=4,
        num_workers=1,
        train_data_pct=0.1,
    )
    
    for batch in dm.val_dataloader():
        print('success')
        break
