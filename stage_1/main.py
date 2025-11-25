import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"
from model.erp import *
from utils.dataset import *
from torch.utils.data import DataLoader
import torch
import pandas as pd
import argparse

import numpy as np
import wandb
import random

import pytorch_lightning as pl
import urllib3

torch.autograd.set_detect_anomaly(True)

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

wandb.login(key='33e22fa1805b15142daccb73d9069d0af6bd5f22', relogin=True, force=True)


def run(args):
    print(args)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    train_df = pd.read_csv(
        os.path.join(args.dataset_path, args.train_df_root),
        parse_dates=["release_date"],
    )
    test_df = pd.read_csv(
        os.path.join(args.dataset_path, args.test_df_root),
        parse_dates=["release_date"],
    )

    img_folder = os.path.join(args.dataset_path, 'images')

    cat_dict = torch.load(os.path.join(args.dataset_path, "category_labels.pt"))
    col_dict = torch.load(os.path.join(args.dataset_path, "color_labels.pt"))
    fab_dict = torch.load(os.path.join(args.dataset_path, "fabric_labels.pt"))


    test_dataset = Visuelle2_cl_val(test_df,
                                    img_folder,
                                    args.img_emb_root,
                                    cat_dict,
                                    col_dict,
                                    fab_dict,
                                    args.test_local_savepath,
                                    main_root = args.main_root,
                                    ref_local_savepath=args.train_local_savepath,
                                    ref_sales_df=train_df)


    test_loader_pred = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = ERP_neighbor_contrastive_learning(
        embedding_dim=512,
        hidden_dim=512,
        cat_dict=cat_dict,
        col_dict=col_dict,
        fab_dict=fab_dict,
        store_num=125,
        lr=0.0001,
        group_size=args.group_size,
        margin_type=args.margin_type,
        gpu_num=args.gpu_num
    )

    trainer = pl.Trainer(accelerator='gpu', devices=[args.gpu_num],)

    best_model_path = 'stage_1/log/pre_trained/stage1_weight.ckpt'
    best_model_weight = torch.load(best_model_path, map_location=f"cuda:{args.gpu_num}")
    model.load_state_dict(best_model_weight['state_dict'])
    model.eval()

    test_pred = trainer.predict(model, dataloaders=test_loader_pred, )
    test_emb = torch.cat([torch.stack(test_pred[:-1]).reshape(-1, 512), test_pred[-1]], axis=0)
    test_pairwise_dist = torch.cdist(test_emb[:-len(train_df)], test_emb[-len(train_df):], p=2)
    test_pred_dist_argsort = torch.argsort(test_pairwise_dist, axis=1)
    np.save('/'.join(best_model_path.split('/')[:-1]) + '/test_pred_dist_argsort.npy', test_pred_dist_argsort)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Zero-Shot-Item-Sales-Forecasting')

    # ------------------------------------------------------------
    # General settings
    # ------------------------------------------------------------
    parser.add_argument('--seed', type=int, default=21,
                        help='Random seed for reproducibility')
    parser.add_argument('--gpu_num', type=int, default=0,
                        help='GPU index to use')
    parser.add_argument('--num_workers', type=int, default=16,
                        help='Number of dataloader workers')
    parser.add_argument('--batch_size', type=int, default=2048,
                        help='Batch size for dataloader')

    # ------------------------------------------------------------
    # Path settings
    # ------------------------------------------------------------
    parser.add_argument('--main_root', type=str, default='..',
                        help='Main project root directory')
    parser.add_argument('--dataset_path', type=str, default='../visuelle2_benchmark',
                        help='Path to Visuelle2 benchmark dataset')

    parser.add_argument('--train_df_root', type=str, default='stfore_train.csv',
                        help='Train dataframe filename')
    parser.add_argument('--test_df_root', type=str, default='stfore_test.csv',
                        help='Test dataframe filename')

    parser.add_argument('--train_local_savepath', type=str, default='visuelle2_train_cl.pt',
                        help='Local save path for cached train embeddings')
    parser.add_argument('--test_local_savepath', type=str, default='visuelle2_test_cl.pt',
                        help='Local save path for cached test embeddings')

    parser.add_argument('--img_emb_root', type=str, default='fclip_img.pkl',
                        help='Precomputed image embedding file')

    # ------------------------------------------------------------
    # ERP neighbor settings
    # ------------------------------------------------------------
    parser.add_argument('--group_size', type=int, default=512,
                        help='Group size for contrastive neighbor sampling')
    parser.add_argument('--margin_type', type=str, default='maxplus',
                        help='Margin type for contrastive loss')
    parser.add_argument('--n_neighbors', type=int, default=10,
                        help='Number of ERP-nearest neighbors')

    # ------------------------------------------------------------
    # Logging / WandB
    # ------------------------------------------------------------
    parser.add_argument('--wandb_entity', type=str, default='ssl_project',
                        help='WandB entity name')
    parser.add_argument('--wandb_proj', type=str, default='visuelle2',
                        help='WandB project name')
    parser.add_argument('--wandb_dir', type=str, default='../',
                        help='WandB logging directory')

    # ------------------------------------------------------------
    # Model identifier
    # ------------------------------------------------------------
    parser.add_argument('--model_type', type=str, default='vis2_cl_erp',
                        help='Model name identifier for saving and WandB')

    args = parser.parse_args()
    run(args)



