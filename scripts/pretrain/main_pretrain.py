from pprint import pprint
import os
from argparse import ArgumentParser, Namespace
import datetime
from dateutil import tz
import random
import numpy as np
import torch
import warnings
from lightning import seed_everything, Trainer
from lightning.pytorch.tuner import Tuner
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from swanlab.integration.pytorch_lightning import SwanLabLogger
from polaris.datasets.pretrain_datamodule import ECGTextDataModule
from polaris.models.merl_model import MERLModel
from polaris.models.ecgfm_model import ECGFMModel
from polaris.models.melp_model import MELPModel
from polaris.paths import ROOT_PATH as REPO_ROOT_DIR
from polaris.paths import DATA_ROOT_PATH

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch.autograd.set_detect_anomaly(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('high')



def main(hparams: Namespace):

    # ------------------------
    # 1 INIT TRAINER
    # ------------------------
    cn_tz = tz.gettz('Asia/Shanghai')
    now = datetime.datetime.now(cn_tz)
    extension = now.strftime("%Y_%m_%d_%H_%M_%S")
    extension = f"polaris_{hparams.model_name}_{extension}"
    ckpt_dir = os.path.join(
        REPO_ROOT_DIR, f"logs/polaris/ckpts/{extension}")
    os.makedirs(ckpt_dir, exist_ok=True)
    if hparams.model_name in ["merl", "melp"]:
        callbacks = [
            LearningRateMonitor(logging_interval="step"),
            ModelCheckpoint(monitor="val/mean_AUROC", dirpath=ckpt_dir,
                            save_last=False, mode="max", save_top_k=2,
                            auto_insert_metric_name=True),
            EarlyStopping(monitor="val/mean_AUROC", min_delta=0,
                        patience=5, verbose=True, mode="max"),
        ]
    # elif hparams.model_name in ["leadfusion", "vqnsp", "heartlang", "ecgfm"]:
    #     callbacks = [
    #         LearningRateMonitor(logging_interval="step"),
    #         ModelCheckpoint(monitor="val/loss", dirpath=ckpt_dir,
    #                         save_last=False, mode="min", save_top_k=2,
    #                         auto_insert_metric_name=True),
    #         EarlyStopping(monitor="val/loss", min_delta=0,
    #                     patience=5, verbose=True, mode="min"),
    #     ]
    # else:
    #     raise NotImplementedError
    logger_dir = os.path.join(REPO_ROOT_DIR, "logs/polaris")
    os.makedirs(logger_dir, exist_ok=True)
    
    # 使用 SwanLab 记录实验
    logger = SwanLabLogger(
        project="polaris",
        experiment_name=extension,
        save_dir=logger_dir
    )
    
    trainer = Trainer(
        max_epochs=hparams.max_epochs,
        accelerator="gpu",
        accumulate_grad_batches=hparams.accumulate_grad_batches,
        devices=hparams.num_devices,
        strategy="ddp_find_unused_parameters_true",
        precision=32 if hparams.model_name == "ecgfm" else "bf16-mixed",
        callbacks=callbacks,
        logger=logger
    )

    # ------------------------
    # 2 INIT LIGHTNING MODEL and lightning datamodule
    # ------------------------
    hparams.exp_log_dir = os.path.join(
        REPO_ROOT_DIR, f"data/{extension}/exp_logs")
    
    if hparams.model_name == "merl":
        datamodule = ECGTextDataModule(
            dataset_dir=str(DATA_ROOT_PATH),
            dataset_list=["mimic-iv-ecg"],
            val_dataset_list=hparams.val_dataset_list,
            batch_size=hparams.batch_size,  
            num_workers=hparams.num_workers,
            train_data_pct=hparams.train_data_pct
        )
        model = MERLModel(**vars(hparams))
    elif hparams.model_name == "melp":
        datamodule = ECGTextDataModule(
            dataset_dir=str(DATA_ROOT_PATH),
            dataset_list=["mimic-iv-ecg"],
            val_dataset_list=hparams.val_dataset_list,
            batch_size=hparams.batch_size,  
            num_workers=hparams.num_workers,
            train_data_pct=hparams.train_data_pct,
            use_rlm=False
        )
        model = MELPModel(**vars(hparams))  
    elif hparams.model_name == "ecgfm":
        datamodule = ECGTextDataModule(
            dataset_dir=str(DATA_ROOT_PATH),
            dataset_list=["mimic-iv-ecg"],
            val_dataset_list=None,
            batch_size=hparams.batch_size,  
            num_workers=hparams.num_workers,
            train_data_pct=hparams.train_data_pct,
            use_cmsc=True,
            use_rlm=True
        )
        model = ECGFMModel(**vars(hparams))
    else:
        raise NotImplementedError

    model.training_steps_per_epoch = len(datamodule.train_dataloader()) // hparams.accumulate_grad_batches // hparams.num_devices
    pprint(vars(hparams))

    # ------------------------
    # 3 START TRAINING
    # ------------------------
    # tuner = Tuner(trainer)
    # Find optimal batch size
    # optimal_batch_size = tuner.scale_batch_size(model=model, datamodule=datamodule, init_val=128,
    #                                             mode="binsearch")
    # datamodule.batch_size = optimal_batch_size
    # print(f"Optimal batch size: {optimal_batch_size}")
    # Find optimal learning rate
    # lr_finder = tuner.lr_find(model=model, datamodule=datamodule, max_lr=1e-3)
    # model.lr = lr_finder.suggestion()
    trainer.fit(model, datamodule=datamodule)


if __name__ == '__main__':
    '''
CUDA_VISIBLE_DEVICES=0,1,2,3 python main_pretrain.py --num_devices 4 --num_workers 12 --train_data_pct 1 \
    --text_encoder_name fuyingw/heart_bert \
    --lr 2e-4 --model_name melp --batch_size 64 --max_epochs 100 \
    --ecg_encoder_name ecgfm \
    --clip_loss_weight 1.0 --caption_loss_weight 2.0 --local_loss_weight 0.2
'''
    parser = ArgumentParser(description="Pretraining Multimodal ECG Foundation Model.")
    parser.add_argument("--model_name", type=str, default="melp",
                        choices=["merl", "ecgfm", "melp"])
    parser.add_argument("--model_size", type=str, default="base")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_data_pct", type=float, default=1.)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--num_devices", type=int, default=1)
    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--accumulate_grad_batches", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--ecg_encoder_name", type=str, default="ecgfm")
    parser.add_argument("--ecg_encoder_weight", type=str, default="")
    parser.add_argument("--text_encoder_name", type=str, default="")
    parser.add_argument("--clip_loss_weight", type=float, default=1.)
    parser.add_argument("--caption_loss_weight", type=float, default=2.0)
    parser.add_argument("--local_loss_weight", type=float, default=0.2)
    parser.add_argument("--n_queries_contrast", type=int, default=12)
    parser.add_argument("--val_dataset_list", type=str, nargs="+", 
                        default=["ptbxl-super", "ptbxl-sub", "ptbxl-form", "ptbxl-rhythm", 
                                  "icbeb", "chapman"])

    hparams = parser.parse_args()

    # set random seed
    random.seed(hparams.seed)
    np.random.seed(hparams.seed)
    torch.manual_seed(hparams.seed)
    torch.cuda.manual_seed(hparams.seed)
    seed_everything(hparams.seed)
    main(hparams)