import torch
import torch.nn as nn
from transformers import CLIPProcessor, CLIPModel


class TimeDistributed(nn.Module):
    # Takes any module and stacks the time dimension with the batch dimenison of inputs before applying the module
    # Insipired from https://keras.io/api/layers/recurrent_layers/time_distributed/
    # https://discuss.pytorch.org/t/any-pytorch-function-can-work-as-keras-timedistributed/1346/4
    def __init__(self, module, batch_first=True):
        super(TimeDistributed, self).__init__()
        self.module = module  # Can be any layer we wish to apply like Linear, Conv etc
        self.batch_first = batch_first

    def forward(self, x):
        if len(x.size()) <= 2:
            return self.module(x)

        # Squash samples and timesteps into a single axis
        x_reshape = x.permute(0, 2, 1).contiguous().view(-1, x.size(1))
        y = self.module(x_reshape)

        # We have to reshape Y
        if self.batch_first:
            y = y.contiguous().view(x.size(0), -1, y.size(
                -1))  # (samples, timesteps, output_size)
        else:
            y = y.view(-1, x.size(1),
                       y.size(-1))  # (timesteps, samples, output_size)

        return y


class FeatureFusionNetwork(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, dropout=0.2):
        super(FeatureFusionNetwork, self).__init__()
        self.batchnorm = nn.BatchNorm1d(embedding_dim * 4)
        self.feature_fusion = nn.Sequential(
            nn.Linear(embedding_dim * 4, hidden_dim * 2, bias=False),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        self.image_ln = nn.LayerNorm(512)
        self.text_ln = nn.LayerNorm(512)
        self.temp_ln = nn.LayerNorm(512)
        self.meta_ln = nn.LayerNorm(512)
        self.feature_ln = nn.LayerNorm(512)


        self.activation = nn.GELU()

    def forward(self, image_embedding, text_embedding, temporal_embedding, meta_embedding):

        image_embedding = self.image_ln(image_embedding)
        text_embedding = self.text_ln(text_embedding)
        temporal_embedding = self.temp_ln(temporal_embedding)
        meta_embedding = self.meta_ln(meta_embedding)

        features = self.activation(torch.cat([image_embedding, text_embedding, temporal_embedding, meta_embedding], dim=1))
        features = self.batchnorm(features)
        features = self.feature_fusion(features)
        features = self.feature_ln(self.activation(features))


        return features


class TemporalFeatureEncoder(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.day_embedding = nn.Linear(1, embedding_dim)
        self.week_embedding = nn.Linear(1, embedding_dim)
        self.month_embedding = nn.Linear(1, embedding_dim)
        self.year_embedding = nn.Linear(1, embedding_dim)
        self.fusion_layer = nn.Linear(embedding_dim * 3, embedding_dim)
        self.dropout = nn.Dropout(0.2)

    def forward(self, temporal_features):
        w = temporal_features[:, 0].unsqueeze(1)
        m = temporal_features[:, 1].unsqueeze(1)
        y = temporal_features[:, 2].unsqueeze(1)
        w_emb, m_emb, y_emb = self.week_embedding(w), self.month_embedding(m), self.year_embedding(y)
        temporal_embeddings = self.fusion_layer(torch.cat([w_emb, m_emb, y_emb], dim=1))
        temporal_embeddings = self.dropout(temporal_embeddings)

        return temporal_embeddings


class AttributeEncoder(nn.Module):
    def __init__(self, num_cat, num_col, num_fab, num_store, embedding_dim):
        super(AttributeEncoder, self).__init__()
        self.cat_embedder = nn.Embedding(num_cat, embedding_dim)
        self.col_embedder = nn.Embedding(num_col, embedding_dim)
        self.fab_embedder = nn.Embedding(num_fab, embedding_dim)
        self.store_embedder = nn.Embedding(num_store, embedding_dim)
        self.fusion_layer = nn.Linear(embedding_dim*4, embedding_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self,  cat, col, fab, store):
        cat_emb = self.dropout(self.cat_embedder(cat))
        col_emb = self.dropout(self.col_embedder(col))
        fab_emb = self.dropout(self.fab_embedder(fab))
        store_emb = self.dropout(self.store_embedder(store))
        attribute_embeddings = self.fusion_layer(torch.cat([cat_emb, col_emb,fab_emb, store_emb], dim=1))

        return attribute_embeddings


class TextEmbedder(nn.Module):
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