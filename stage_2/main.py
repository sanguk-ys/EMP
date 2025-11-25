import os
import argparse
import wandb
import torch
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from datetime import datetime
from models.forecast_fourier import EMP
from torch.utils.data import DataLoader
from utils.scaling_method import Fourier_transform_pos
from utils.dataset import Visuelle2

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ['CURL_CA_BUNDLE'] = ''
torch.autograd.set_detect_anomaly(True)


def run(args):
    print(args)
    pl.seed_everything(args.seed)

    # ---------------- Load CSV ----------------
    train_df = pd.read_csv(
        os.path.join(args.dataset_path, args.train_df_root),
        parse_dates=["release_date"],
    )
    test_df = pd.read_csv(
        os.path.join(args.dataset_path, args.test_df_root),
        parse_dates=["release_date"],
    )

    # Fourier Mapping
    sales_transform = Fourier_transform_pos(args.f_max_len, args.hidden_dim, args.fourier_B)

    # ---------------- Load attribute encodings ----------------
    cat_dict = torch.load(os.path.join(args.dataset_path, "category_labels.pt"))
    col_dict = torch.load(os.path.join(args.dataset_path, "color_labels.pt"))
    fab_dict = torch.load(os.path.join(args.dataset_path, "fabric_labels.pt"))

    # ---------------- Load Google trends ----------------
    gtrends = pd.read_csv(
        os.path.join(args.dataset_path, "vis2_gtrends_data.csv"),
        index_col=[0], parse_dates=True
    )

    img_folder = os.path.join(args.dataset_path, 'images')

    # ===============================================================
    # Google Trends and POP features are computed and stored in the dataset
    # Nevertheless, the EMP model architecture
    # does not take these features as input.
    # They remain available for future versions of the model but do not
    # contribute to the present inference pipeline.
    # ===============================================================
    testset = Visuelle2(
        sales_df=test_df,
        img_root=img_folder,
        img_emb_root=args.img_emb_root,
        gtrends=gtrends,
        cat_dict=cat_dict,
        col_dict=col_dict,
        fab_dict=fab_dict,
        sales_transform=sales_transform,
        pop_root=args.pop_root,
        distance_sorted_root=args.test_distance_sorted,
        n_neighbors=args.n_neighbors,
        ref_sales_df=train_df,
        use_gtrends=bool(args.use_gtrends),
        use_pop=bool(args.use_pop),
        use_kitem=bool(args.use_kitem),
    )

    test_loader = DataLoader(testset, batch_size=2048, shuffle=False, num_workers=4)

    model = EMP(
        hidden_dim=args.hidden_dim,
        output_dim=args.output_dim,
        cat_dict=cat_dict,
        col_dict=col_dict,
        fab_dict=fab_dict,
        num_heads=args.num_attn_heads,
        num_layers=args.num_hidden_layers,
        gpu_num=args.gpu_num,
        lr=args.learning_rate,
        sales_transform=sales_transform,
        n_neighbors=args.n_neighbors,
    )

    dt_string = datetime.now().strftime("%Y%m%d-%H%M")[2:]
    model_savename = dt_string + '_' + args.model_type

    wandb.init(
        entity=args.wandb_entity,
        project=args.wandb_proj,
        name=model_savename,
        dir=args.wandb_dir
    )
    wandb_logger = WandbLogger(name=model_savename)

    trainer = pl.Trainer(
        gpus=[args.gpu_num],
        max_epochs=args.epochs,
        check_val_every_n_epoch=1,
        logger=wandb_logger,
    )

    best_model_weight = torch.load(
        '/home/sflab/SFLAB/su_vis2/EMP/log/251125-0950_EMP_vis2_for_git_w_storenum/---epoch=7---.ckpt'
    )
    model.load_state_dict(best_model_weight['state_dict'])
    model.eval()

    trainer.test(model, dataloaders=test_loader)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Zero-shot sales forecasting')

    # General
    parser.add_argument('--seed', type=int, default=21)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--gpu_num', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=128)

    # Model specific
    parser.add_argument('--hidden_dim', type=int, default=512)
    parser.add_argument('--output_dim', type=int, default=12)
    parser.add_argument('--num_attn_heads', type=int, default=8)
    parser.add_argument('--num_hidden_layers', type=int, default=2)
    parser.add_argument('--learning_rate', type=float, default=0.0001)
    parser.add_argument('--fourier_B', type=int, default=10000)
    parser.add_argument('--n_neighbors', type=int, default=10)

    # Contextual signal flags
    parser.add_argument('--use_gtrends', type=int, default=0)
    parser.add_argument('--use_pop', type=int, default=0)
    parser.add_argument('--use_kitem', type=int, default=1)

    # Paths
    parser.add_argument('--dataset_path', type=str, default='../visuelle2_benchmark/')
    parser.add_argument('--train_df_root', type=str, default='stfore_train.csv')
    parser.add_argument('--test_df_root', type=str, default='stfore_test.csv')
    parser.add_argument('--test_distance_sorted', type=str, default='../test_pred_dist_argsort.npy')

    parser.add_argument('--img_emb_root', type=str, default='../fclip_img.pkl')
    parser.add_argument('--pop_root', type=str, default='../pop_fill_mean.pickle')

    # Model identity
    parser.add_argument('--model_type', type=str, default='EMP_vis2')
    parser.add_argument('--f_max_len', type=int, default=751 + 1)

    # wandb
    parser.add_argument('--wandb_entity', type=str, default='ssl_project')
    parser.add_argument('--wandb_proj', type=str, default='visuelle2')
    parser.add_argument('--wandb_dir', type=str, default='../')

    args = parser.parse_args()
    run(args)
