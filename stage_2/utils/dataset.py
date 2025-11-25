import os
import torch
import pandas as pd
import numpy as np
import pickle
from tqdm import tqdm
from PIL import ImageFile
from torch.utils.data import TensorDataset
from sklearn.preprocessing import MinMaxScaler
from utils.timefeatures import time_features

ImageFile.LOAD_TRUNCATED_IMAGES = True


class Visuelle2:
    def __init__(
            self,
            sales_df,
            img_root,
            img_emb_root,
            gtrends,
            cat_dict,
            col_dict,
            fab_dict,
            sales_transform,
            pop_root,
            distance_sorted_root,
            n_neighbors,
            ref_sales_df,
            use_gtrends=False,
            use_pop=False,
            use_kitem=True,
    ):
        self.sales_df = sales_df
        self.img_emb_root = img_emb_root
        self.gtrends = gtrends
        self.cat_dict = cat_dict
        self.col_dict = col_dict
        self.fab_dict = fab_dict
        self.img_root = img_root
        self.sales_transform = sales_transform
        self.pop_root = pop_root
        self.distance_sorted_root = distance_sorted_root
        self.n_neighbors = n_neighbors
        self.ref_sales_df = ref_sales_df
        self.use_gtrends = use_gtrends
        self.use_pop = use_pop
        self.use_kitem = use_kitem

        self.__read_data__()

        print("Processing dataset (no caching)...")
        self.dataset = self.preprocess_data()
        print("Done.")

    def __read_data__(self):
        self.img_emb_dict = pickle.load(open(self.img_emb_root, 'rb'))
        self.pop_signal_dict = pickle.load(open(self.pop_root, 'rb'))

        self.distance_sorted_mat = np.load(os.path.join('/home/sflab/SFLAB/su_vis2/data/', self.distance_sorted_root))

        self.memory_sales_df = self.ref_sales_df.iloc[:, -12:]

    def preprocess_data(self):
        ts_f_mapping_list = []
        gtrends_list = []
        pop_signal_list = []
        pt_img_list = []
        k_item_sales_list_all = []
        temporal_list = []
        real_sales_list = []

        for idx, row in tqdm(self.sales_df.iterrows(), total=len(self.sales_df)):

            # =======================================
            # Fourier mapping of last 12 sales
            # =======================================
            ts_f_mapping = self.sales_transform.transform(
                torch.tensor(row[-12:].values.astype(int), dtype=torch.int64)
            )

            cat = row.category
            col = row.color
            fab = row.fabric
            release_date = row.release_date

            # =======================================
            # Google Trends (ON/OFF) default OFF
            # =======================================
            if self.use_gtrends:
                gtrend_start = release_date - pd.DateOffset(weeks=52)

                def _extract(col_name):
                    seq = self.gtrends.loc[gtrend_start:release_date][col_name][-52:].values
                    if len(seq) < 52:
                        seq = self.gtrends.loc[:release_date][col_name][-52:].values

                    seq = MinMaxScaler().fit_transform(seq.reshape(-1, 1)).flatten()

                    if len(seq) < 52:
                        seq = np.pad(seq, (52 - len(seq), 0), 'constant')

                    return seq[:52]

                g_var = torch.FloatTensor([
                    _extract(cat),
                    _extract(col),
                    _extract(fab),
                ])
            else:
                g_var = torch.zeros(3, 52)

            # =======================================
            # POP signal (ON/OFF) default OFF
            # =======================================
            if self.use_pop and row['image_path'] in self.pop_signal_dict:
                pop_signal = torch.FloatTensor(self.pop_signal_dict[row['image_path']])
            else:
                pop_signal = torch.zeros(52)

            # =======================================
            # Image embedding
            # =======================================
            try:
                pt_img = torch.FloatTensor(self.img_emb_dict[row['image_path']]).squeeze()
            except:
                pt_img = torch.zeros(512)

            # =======================================
            # Temporal features
            # =======================================
            rd = pd.DatetimeIndex([release_date])
            temporal_feature = torch.FloatTensor([
                time_features(rd, freq='w')[0][0],
                time_features(rd, freq='m')[0][0],
                time_features(rd, freq='y')[0][0],
            ])

            # ==============================================
            # K nearest reference items (ON/OFF) default ON
            # ==============================================
            if self.use_kitem:
                k_nearest_idx = self.distance_sorted_mat[idx][:self.n_neighbors]
                k_item_sales_list = []

                for k_idx in k_nearest_idx:
                    mrow = self.memory_sales_df.iloc[k_idx]
                    mapped = self.sales_transform.transform(
                        torch.tensor(mrow.values.astype(int), dtype=torch.int64)
                    )
                    k_item_sales_list.append(mapped)

                k_item_sales = torch.stack(k_item_sales_list).reshape(-1, 512)
            else:
                k_item_sales = torch.zeros(self.n_neighbors, 512)

            # =======================================
            # Ground-truth real sales
            # =======================================
            real_sales = torch.FloatTensor(row[-12:].values)

            # Accumulate
            ts_f_mapping_list.append(ts_f_mapping)
            gtrends_list.append(g_var)
            pop_signal_list.append(pop_signal)
            pt_img_list.append(pt_img)
            k_item_sales_list_all.append(k_item_sales)
            temporal_list.append(temporal_feature)
            real_sales_list.append(real_sales)

        # =======================================
        # Category / Color / Fabric indices
        # =======================================
        categories = torch.LongTensor([self.cat_dict[c] for c in self.sales_df["category"]])
        colors = torch.LongTensor([self.col_dict[c] for c in self.sales_df["color"]])
        fabrics = torch.LongTensor([self.fab_dict[c] for c in self.sales_df["fabric"]])

        # =======================================
        # TensorDataset creation
        # =======================================
        return TensorDataset(
            torch.stack(ts_f_mapping_list),     # 0
            categories,                         # 1
            colors,                             # 2
            fabrics,                            # 3
            torch.stack(temporal_list),         # 4
            torch.stack(gtrends_list),          # 5
            torch.stack(pop_signal_list),       # 6
            torch.stack(pt_img_list),           # 7
            torch.stack(k_item_sales_list_all), # 8
            torch.stack(real_sales_list),       # 9
        )

    def __getitem__(self, idx):
        return self.dataset[idx]

    def __len__(self):
        return len(self.dataset)