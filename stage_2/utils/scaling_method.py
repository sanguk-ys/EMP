import math
import torch


class Fourier_transform_pos():
    def __init__(self, f_max_len, hidden_dim, b_scale):
        f_pe = torch.zeros(f_max_len, hidden_dim)
        position = torch.arange(0, f_max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hidden_dim, 2).float() * (-math.log(float(b_scale)) / hidden_dim))
        f_pe[:, 0::2] = torch.sin(position * div_term)
        f_pe[:, 1::2] = torch.cos(position * div_term)
        self.f_pe = f_pe
        self.div_term = div_term

    def transform(self, sales):
        transformed_data = torch.index_select(self.f_pe, 0, sales)
        return transformed_data

    def inverse_transform(self, emb_sales):
        sales = torch.argmax(torch.tensordot(self.f_pe, emb_sales, dims=([1], [2])), dim=0)
        return sales

class Normalization_max_sales():
    def __init__(self, max_sales):
        self.max_sales = max_sales

    def transform(self, sales):
        transformed_data = sales / self.max_sales
        return transformed_data

    def inverse_transform(self, unscaled_sales):
        sales = unscaled_sales * self.max_sales
        return sales


