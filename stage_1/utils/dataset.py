import os
import torch
import pickle
import pandas as pd
from PIL import ImageFile
from utils.timefeatures import time_features
from torchvision.transforms import Resize, ToTensor, Normalize, Compose

ImageFile.LOAD_TRUNCATED_IMAGES = True


class Visuelle2_cl_val():
    """
    Dataset for Stage 1 ERP Contrastive Learning Validation / Prediction.
    Loads:
        - item metadata (category, color, fabric)
        - image embeddings (from Fashion-CLIP) https://huggingface.co/patrickjohncyh/fashion-clip
        - time features (week/month/year)
        - reference items (memory bank) appended after test items
    """

    def __init__(
        self,
        sales_df,
        img_folder,
        img_emb_root,
        cat_dict,
        col_dict,
        fab_dict,
        local_savepath,
        main_root='/home/workspace/',
        ref_local_savepath=None,
        ref_sales_df=None,
    ):
        """
        Args:
            sales_df (pd.DataFrame):
                Test dataframe containing items to embed.
            img_folder (str):
                Directory containing original product images.
            img_emb_root (str):
                Path to precomputed image-embedding file (Fashion-CLIP).
            cat_dict, col_dict, fab_dict (dict):
                Encoders for categorical attributes.
            local_savepath (str):
                Cache path (usually unused here).
            main_root (str):
                Main project directory root.
            ref_local_savepath (str):
                Path to stored reference embeddings (unused here).
            ref_sales_df (pd.DataFrame):
                Reference items (train_df) used to construct a memory bank.
                After all test items, ref items are appended for pairwise distance calc.
        """

        self.sales_df = sales_df
        self.prepo_data_folder = main_root + "../visuelle2_benchmark/"

        # Original images folder (not used here but kept for compatibility)
        self.img_folder = img_folder

        # Load precomputed F-CLIP image embeddings
        self.img_emb_dict = pickle.load(open(os.path.join("..", img_emb_root), 'rb'))

        # Metadata dictionaries
        self.cat_dict = cat_dict
        self.col_dict = col_dict
        self.fab_dict = fab_dict

        self.main_root = main_root
        self.ref_local_savepath = ref_local_savepath

        # Image transform (if raw image loading were used)
        self.img_transforms = Compose([
            Resize((299, 299)),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406],
                      std=[0.229, 0.224, 0.225]),
        ])

        # Reference dataframe (train_df)
        self.memory_sales_df = ref_sales_df


    def __getitem__(self, idx):
        """
        Returns attribute + time + image embedding for:
            - test item (if idx < len(test))
            - reference item (if idx >= len(test))
        """

        # --- Select test item or reference item ---
        if idx < len(self.sales_df):
            row = self.sales_df.iloc[idx]
        else:
            ref_idx = idx - len(self.sales_df)
            row = self.memory_sales_df.iloc[ref_idx]

        # --- Image embedding ---
        # If image path missing, return zero vector embedding
        image_path = row["image_path"]
        try:
            pt_img = torch.FloatTensor(self.img_emb_dict[image_path]).squeeze()
        except:
            pt_img = torch.zeros(512)  # fallback when missing

        # --- Time features (week, month, year) ---
        release_date = pd.DatetimeIndex([row["release_date"]])
        temporal_features = torch.FloatTensor([
            time_features(release_date, freq='w')[0][0],
            time_features(release_date, freq='m')[0][0],
            time_features(release_date, freq='y')[0][0],
        ])

        # --- Categorical attributes ---
        categories = self.cat_dict[row['category']]
        colors = self.col_dict[row['color']]
        fabrics = self.fab_dict[row['fabric']]
        stores = 0  # no store info in Visuelle2 CL stage

        categories, colors, fabrics, stores = torch.LongTensor(
            [categories, colors, fabrics, stores]
        )

        return categories, colors, fabrics, stores, temporal_features, pt_img


    def __len__(self):
        """
        Total length = test items + reference items
        This allows ERP distance computation by comparing every test item
        to every reference item.
        """
        return len(self.sales_df) + len(self.memory_sales_df)
