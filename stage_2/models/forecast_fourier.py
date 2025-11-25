import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics.regression import SymmetricMeanAbsolutePercentageError
from transformers import CLIPProcessor, CLIPModel

torch.set_printoptions(precision=4)


class StaticFeatureEncoder(nn.Module):
    """
    Encodes static item-level features:
    - image embedding (FCLIP)
    - text embedding (FCLIP text)
    - temporal dummy encoding
    - categorical metadata (category/color/fabric)

    All four features are layer-normalized, fused, and projected into a shared hidden_dim.
    This produces a unified static representation for the decoder.
    """
    def __init__(self, hidden_dim, dropout=0.2):
        super(StaticFeatureEncoder, self).__init__()

        self.batchnorm = nn.BatchNorm1d(hidden_dim * 4)

        self.feature_fusion = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim * 2, bias=False),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )

        self.meta_linear = nn.Linear(51, hidden_dim)

        self.image_ln = nn.LayerNorm(512)
        self.text_ln = nn.LayerNorm(512)
        self.temp_ln = nn.LayerNorm(512)
        self.meta_ln = nn.LayerNorm(512)
        self.feature_ln = nn.LayerNorm(512)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, img_encoding, text_encoding, temporal_encoding=None, meta_data=None):
        image_embedding = self.dropout(self.image_ln(img_encoding))
        text_embedding = self.dropout(self.text_ln(text_encoding))
        temporal_embedding = self.dropout(self.temp_ln(temporal_encoding))
        meta_embedding = self.dropout(self.meta_ln(meta_data))

        features = self.activation(torch.cat([image_embedding, text_embedding, temporal_embedding, meta_embedding], dim=1))
        features = self.batchnorm(features)
        features = self.feature_fusion(features)
        features = self.feature_ln(self.activation(features))

        return features

class K_item_sales_Embedder(nn.Module):
    """
    Encodes contextual signals from:
      (1) Google Trends(zero input as default)    →  [batch, 3, trend_len]
      (2) POP signals(zero input as default)      →  [batch, trend_len]
      (3) k-nearest reference item sales (ERP-retrieved)

    Each channel is linearly projected and gated.
    The concatenated sequence is passed through a Transformer encoder,
    producing contextual memory used by the decoder.

    pop_signal and gtrends may be zero-tensors depending on user flags.
    """
    def __init__(self, embedding_dim, trend_len, num_trends):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.trend_len = trend_len

        # Input Linear Layers
        self.input_linear_gtr = nn.Linear(num_trends, embedding_dim)
        self.input_linear_pop = nn.Linear(1, embedding_dim)

        # Gating Layers
        self.gtrend_gate = nn.Sequential(nn.Linear(embedding_dim, embedding_dim), nn.Sigmoid())
        self.k_item_sales_gate = nn.Sequential(nn.Linear(embedding_dim, embedding_dim), nn.Sigmoid())
        self.pop_gate = nn.Sequential(nn.Linear(embedding_dim, embedding_dim), nn.Sigmoid())

        # Dropout
        self.dropout = nn.Dropout(0.2)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=4, dropout=0.2)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)

    def forward(self, gtrends, k_item_sales, mask, pop_signal=None):
        gtrend_emb = self.input_linear_gtr(gtrends)  # Linear projection
        gtrend_emb = self.gtrend_gate(gtrend_emb) * gtrend_emb  # Apply gating
        gtrend_emb = self.dropout(gtrend_emb)


        k_item_sales_emb = self.k_item_sales_gate(k_item_sales) * k_item_sales
        k_item_sales_emb = self.dropout(k_item_sales_emb)

        pop_emb = self.input_linear_pop(pop_signal.unsqueeze(2))
        pop_emb = self.pop_gate(pop_emb) * pop_emb
        pop_emb = self.dropout(pop_emb)

        # Combine embeddings
        combined = torch.cat([gtrend_emb, pop_emb, k_item_sales_emb], axis=1)

        # Pass through Transformer Encoder with mask
        context_emb = self.encoder(combined.permute(1, 0, 2), mask=mask)
        return context_emb

class DummyEmbedder(nn.Module):
    """
    Temporal feature encoder.
    Uses simple linear projections of (week, month, year)
    followed by fusion and dropout.
    """
    def __init__(self, embedding_dim):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.week_embedding = nn.Linear(1, embedding_dim)
        self.month_embedding = nn.Linear(1, embedding_dim)
        self.year_embedding = nn.Linear(1, embedding_dim)
        self.dummy_fusion = nn.Linear(embedding_dim * 3, embedding_dim)
        self.dropout = nn.Dropout(0.2)

    def forward(self, temporal_features):
        # Temporal dummy variables (week, month, year)
        w, m, y = temporal_features[:, 0].unsqueeze(1), temporal_features[:, 1].unsqueeze(1), temporal_features[:, 2].unsqueeze(1)
        w_emb, m_emb, y_emb = self.week_embedding(w), self.month_embedding(m), self.year_embedding(y)
        temporal_embeddings = self.dummy_fusion(torch.cat([w_emb, m_emb, y_emb], dim=1))
        temporal_embeddings = self.dropout(temporal_embeddings)

        return temporal_embeddings


class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super(TransformerDecoderLayer, self).__init__()

        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        # Implementation of Feedforward models
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = F.relu

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(TransformerDecoderLayer, self).__setstate__(state)

    def forward(self, tgt, memory, tgt_mask, memory_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        tgt2, attn_weights = self.multihead_attn(tgt, memory, memory)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt


class Given_0_nonauto_linear(nn.Module):
    """
    Final projection for non-autoregressive forecasting.
    Takes decoder memory and directly outputs hidden_dim × output_len prediction matrix.
    """
    def __init__(self, hidden_dim, trend_len, output_len):
        super().__init__()

        self.linear1 = nn.Linear(hidden_dim, output_len * hidden_dim, bias=True)

        self.activation = nn.Tanh()

        self.dropout = nn.Dropout(0.2)

    def forward(self, memory_of_decoder_se):
        out = self.activation(self.linear1(memory_of_decoder_se))
        return out

class TextEmbedder(nn.Module):
    """
    Generates text embeddings using Fashion-CLIP.

    Input:
      category, color, fabric IDs → textual prompt "color fabric category"

    Output:
      512-dim CLIP text embedding → projected to hidden_dim
    """

    def __init__(self, embedding_dim, cat_dict, col_dict, fab_dict, gpu_num):
        super().__init__()
        self.fclip = CLIPModel.from_pretrained("patrickjohncyh/fashion-clip")
        self.max_sequence_length = self.fclip.config.text_config.max_position_embeddings
        self.fclip_processor = CLIPProcessor.from_pretrained("patrickjohncyh/fashion-clip")
        self.fclip_tokenizer = self.fclip_processor.tokenizer
        self.embedding_dim = embedding_dim

        self.gpu_num = gpu_num

        self.cat_dict = {v: k for k, v in cat_dict.items()}
        self.col_dict = {v: k for k, v in col_dict.items()}
        self.fab_dict = {v: k for k, v in fab_dict.items()}

        self.fc = nn.Linear(512, embedding_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, category, color, fabric):
        textual_description = [self.col_dict[color.detach().cpu().numpy().tolist()[i]] + ' ' \
                               + self.fab_dict[fabric.detach().cpu().numpy().tolist()[i]] + ' ' \
                               + self.cat_dict[category.detach().cpu().numpy().tolist()[i]] for i in range(len(category))]

        inputs = self.fclip_tokenizer(textual_description, return_tensors="pt", max_length=self.max_sequence_length, padding=True, truncation=True).to('cuda:' + str(self.gpu_num))
        text_embeddings = self.fclip.get_text_features(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])


        return text_embeddings

class AttributeEncoder(nn.Module):
    """
    Embeds categorical attributes:
      - category
      - color
      - fabric
      - (optional) store_id if extended

    Outputs a fused hidden_dim attribute representation.
    """

    def __init__(self, num_cat, num_col, num_fab, embedding_dim):
        super(AttributeEncoder, self).__init__()
        self.cat_embedder = nn.Embedding(num_cat, embedding_dim)
        self.col_embedder = nn.Embedding(num_col, embedding_dim)
        self.fab_embedder = nn.Embedding(num_fab, embedding_dim)
        self.store_embedder = nn.Embedding(126, embedding_dim)
        self.fusion_layer = nn.Linear(embedding_dim * 3, embedding_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, cat, col, fab):
        cat_emb = self.dropout(self.cat_embedder(cat))
        col_emb = self.dropout(self.col_embedder(col))
        fab_emb = self.dropout(self.fab_embedder(fab))

        attribute_embeddings = self.fusion_layer(torch.cat([cat_emb, col_emb, fab_emb], dim=1))

        return attribute_embeddings

class EMP(pl.LightningModule):
    """
    Enhanced Multi-modal Predictor (EMP)

    Components:
      - StaticFeatureEncoder: image + text + temporal + metadata
      - DummyEmbedder: temporal encoding
      - K_item_sales_Embedder: contextual memory from gtrends(zero input as default)/pop(zero input as default)/k-nearest sales
      - TransformerDecoder: static → contextual attention
      - Given_0_nonauto_linear: final forecast head
      - Fourier-based scaling/inverse-scaling for stable training

    Forward pipeline:
      1) Encode temporal variables
      2) Encode image/text/meta static features
      3) Build contextual memory using:
            (gtrends(zero input as default), pop_signal(zero input as default), k_item_sales)
      4) Run Transformer decoder (static → contextual)
      5) Produce 12-week non-autoregressive forecasts
    """

    def __init__(self, hidden_dim, output_dim,
                 cat_dict,
                 col_dict,
                 fab_dict,
                 num_heads,
                 num_layers, gpu_num, lr,
                 sales_transform,
                 n_neighbors,
                 trend_len=52,
                 num_trends=3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.output_len = output_dim

        self.gpu_num = gpu_num
        self.save_hyperparameters()
        self.trend_len = trend_len
        self.lr = lr
        self.n_neighbors = n_neighbors

        self.text_encoder = TextEmbedder(hidden_dim, cat_dict, col_dict, fab_dict, gpu_num)
        self.attribute_encoder = AttributeEncoder(
            len(cat_dict) + 1,
            len(col_dict) + 1,
            len(fab_dict) + 1,
            hidden_dim,
        )

        self.sales_transform = sales_transform

        # Encoder
        self.dummy_encoder = DummyEmbedder(hidden_dim)
        self.k_item_sales_encoder = K_item_sales_Embedder(hidden_dim, trend_len, num_trends)
        self.static_feature_encoder = StaticFeatureEncoder(hidden_dim)

        decoder_layer = TransformerDecoderLayer(d_model=self.hidden_dim, nhead=num_heads, dim_feedforward=self.hidden_dim * 4,
                                                            dropout=0.1)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers)
        self.given_0_nonauto_linear = Given_0_nonauto_linear(hidden_dim, trend_len, output_dim)

    def _generate_deocder_fisrt_mask(self):
        mask = (torch.triu(torch.ones(self.trend_len, self.trend_len)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0)).to('cuda:' + str(self.gpu_num))

        return mask

    def _generate_k_item_sales_mask(self):
        mask = (torch.triu(torch.ones(self.output_len, self.output_len)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0)).to('cuda:' + str(self.gpu_num))
        mask_list = []
        for i in range(self.n_neighbors):
            mask_list.append(mask)
        column_mask = torch.stack(mask_list).reshape(self.output_len * self.n_neighbors, self.output_len)
        column_mask_list = []
        for i in range(self.n_neighbors):
            column_mask_list.append(column_mask)

        mask = torch.stack(column_mask_list, axis=1).reshape(self.output_len * self.n_neighbors, self.output_len * self.n_neighbors)
        return mask

    def forward(self, categories, colors, fabrics, temporal_features, gtrends, pop_signal, images, k_item_sales):
        temporal_encoding = self.dummy_encoder(temporal_features)

        img_encoding = images
        text_encoding = self.text_encoder(categories, colors, fabrics)
        meta_data = self.attribute_encoder(categories, colors, fabrics)

        static_feature_fusion = self.static_feature_encoder(img_encoding, text_encoding, temporal_encoding=temporal_encoding, meta_data=meta_data)

        mask_0 = torch.full((self.trend_len, self.trend_len), float(0.0)).to(f'cuda:{self.gpu_num}')
        mask_1 = torch.full((self.trend_len, self.n_neighbors * 12), float(0.0)).to(f'cuda:{self.gpu_num}')
        gtrends_mask = torch.cat([self._generate_deocder_fisrt_mask(), mask_0, mask_1], axis=1)
        pop_mask = torch.cat([mask_0, self._generate_deocder_fisrt_mask(), mask_1], axis=1)
        k_item_mask = torch.cat([mask_1.transpose(1, 0), mask_1.transpose(1, 0), self._generate_k_item_sales_mask()], axis=1)

        contextual_emb = self.k_item_sales_encoder(gtrends.permute(0, 2, 1), k_item_sales, torch.cat([gtrends_mask, pop_mask, k_item_mask], axis=0), pop_signal=pop_signal)

        memory_of_decoder = self.decoder(tgt=static_feature_fusion.unsqueeze(0), memory=contextual_emb)
        forecast = self.given_0_nonauto_linear(memory_of_decoder).reshape(-1, self.output_len, 512)

        return forecast

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_week_ad_smape',
                'interval': 'epoch',
                'frequency': 1,
            }
        }

    def training_step(self, train_batch, batch_idx):
        item_sales, categories, colors, fabrics, stores, \
            temporal_features, gtrends, pop_signal, images, k_item_sales, real_value_sales = train_batch

        forecasted_sales = self.forward(categories, colors, fabrics, temporal_features, gtrends, pop_signal, images, k_item_sales)
        loss = F.mse_loss(item_sales, forecasted_sales)

        self.log('train_loss_total', loss)

        with torch.no_grad():
            forecasted_sales = self.forward(categories, colors, fabrics, temporal_features, gtrends, pop_signal, images, k_item_sales)

            unscaled_forecasted_sales = self.sales_transform.inverse_transform(forecasted_sales.detach().cpu())

            gt = real_value_sales
            pred = unscaled_forecasted_sales

            ad_smape = SymmetricMeanAbsolutePercentageError()
            smape_adjust_gs_stack = torch.stack([ad_smape(pred[i], gt.detach().cpu()[i]) * 0.5 for i in range(len(gt))])
            smape_adjust_gs = torch.mean(smape_adjust_gs_stack)

            wape = torch.sum(torch.abs(gt.detach().cpu() - pred)) / torch.sum(gt.detach().cpu())

            mae_stack = F.l1_loss(gt.detach().cpu(), pred, reduction='none').mean(axis=-1)
            mae = torch.mean(mae_stack)

            self.log('train_week_ad_smape', smape_adjust_gs)
            self.log('train_week_wape', wape)
            self.log('train_week_mae', mae)

            print('train_week_ad_smape', smape_adjust_gs,
                  'train_week_wape', wape,
                  'train_week_mae', mae,
                  'LR:', self.optimizers().param_groups[0]['lr'],
                  )

        return loss

    def validation_step(self, valid_batch, batch_idx):
        item_sales, categories, colors, fabrics, stores, \
            temporal_features, gtrends, pop_signal, images, k_item_sales, real_value_sales = valid_batch

        forecasted_sales = self.forward(categories, colors, fabrics, temporal_features, gtrends, pop_signal, images, k_item_sales)

        unscaled_forecasted_sales = self.sales_transform.inverse_transform(forecasted_sales.detach().cpu())

        gt = real_value_sales
        pred = unscaled_forecasted_sales

        ad_smape = SymmetricMeanAbsolutePercentageError()
        smape_adjust_gs_stack = torch.stack([ad_smape(pred[i], gt.detach().cpu()[i]) * 0.5 for i in range(len(gt))])
        smape_adjust_gs = torch.mean(smape_adjust_gs_stack)

        wape = torch.sum(torch.abs(gt.detach().cpu() - pred)) / torch.sum(gt.detach().cpu())

        mae_stack = F.l1_loss(gt.detach().cpu(), pred, reduction='none').mean(axis=-1)
        mae = torch.mean(mae_stack)

        self.log('val_week_ad_smape', smape_adjust_gs)
        self.log('val_week_wape', wape)
        self.log('val_week_mae', mae)

        print('val_week_ad_smape', smape_adjust_gs,
              'val_week_wape', wape,
              'val_week_mae', mae,
              )


    def test_step(self, test_batch, batch_idx):
        item_sales, categories, colors, fabrics, stores, \
            temporal_features, gtrends, pop_signal, images, k_item_sales, real_value_sales = test_batch

        forecasted_sales = self.forward(categories, colors, fabrics, temporal_features, gtrends, pop_signal, images, k_item_sales)

        unscaled_forecasted_sales = self.sales_transform.inverse_transform(forecasted_sales.detach().cpu())

        gt = real_value_sales
        pred = unscaled_forecasted_sales

        ad_smape = SymmetricMeanAbsolutePercentageError()
        smape_adjust_gs_stack = torch.stack([ad_smape(pred[i], gt.detach().cpu()[i]) * 0.5 for i in range(len(gt))])
        smape_adjust_gs = torch.mean(smape_adjust_gs_stack)

        wape = torch.sum(torch.abs(gt.detach().cpu() - pred)) / torch.sum(gt.detach().cpu())

        mae_stack = F.l1_loss(gt.detach().cpu(), pred, reduction='none').mean(axis=-1)
        mae = torch.mean(mae_stack)

        self.log('test_week_ad_smape', smape_adjust_gs)
        self.log('test_week_wape', wape)
        self.log('test_week_mae', mae)

        print('test_week_ad_smape', smape_adjust_gs,
              'test_week_wape', wape,
              'test_week_mae', mae,
              )

    def predict_step(self, test_batch, batch_idx):
        item_sales, categories, colors, fabrics, stores, \
            temporal_features, gtrends, pop_signal, images, k_item_sales, real_value_sales = test_batch

        forecasted_sales = self.forward(categories, colors, fabrics, temporal_features, gtrends, pop_signal, images, k_item_sales)

        unscaled_forecasted_sales = self.sales_transform.inverse_transform(forecasted_sales.detach().cpu())

        pred = unscaled_forecasted_sales

        return pred
