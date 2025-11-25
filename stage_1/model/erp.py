import torch
from model.layer import FeatureFusionNetwork, TextEmbedder, TemporalFeatureEncoder, AttributeEncoder
from model.base import PytorchLightningBase
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


class ERP_neighbor_contrastive_learning_predictor(PytorchLightningBase):
    """
    Stage 1: ERP-based Neighbor Contrastive Learning (Prediction Mode)
    ---------------------------------------------------------------
    This class handles:
        - forward embedding computation
        - triplet-style hard positive / hard negative selection
        - loss computation for train/val phases
        - inference-time embedding extraction

    """

    def __init__(
        self,
        embedding_dim,
        hidden_dim,
        cat_dict,
        col_dict,
        fab_dict,
        store_num,
        lr,
        group_size,
        margin_type,
        gpu_num=0,
        n_neighbors=10,
    ):
        """
        Args:
            embedding_dim (int): Dimensionality of output item embedding.
            hidden_dim (int): Hidden dimension for fusion network.
            group_size (int): Size of each group for anchor-positive-negative structure.
                             Within one group:
                                - index 0 is anchor
                                - next n_neighbors are positives
                                - remaining are negatives
            margin_type (str): {'maxplus', 'softplus'} margin formulation.
            n_neighbors (int): Number of ERP-nearest positive items per group.
        """

        super().__init__()
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.lr = lr
        self.group_size = group_size
        self.margin_type = margin_type
        self.n_neighbors = n_neighbors
        self.save_hyperparameters()

        # Encoders for different modalities
        self.feature_fusion_network = FeatureFusionNetwork(embedding_dim, hidden_dim)
        self.text_encoder = TextEmbedder(embedding_dim, cat_dict, col_dict, fab_dict, gpu_num)
        self.temp_encoder = TemporalFeatureEncoder(embedding_dim)
        self.attribute_encoder = AttributeEncoder(
            len(cat_dict) + 1,
            len(col_dict) + 1,
            len(fab_dict) + 1,
            store_num + 1,
            embedding_dim,
        )

    # ---------------------------------------------------------------------
    # Shared step logic for train/valid
    # ---------------------------------------------------------------------
    def phase_step(self, batch, phase):
        """
        Compute hardest-positive / hardest-negative contrastive loss.

        Loss structure per group:
            anchor = first element of each group (group_id * group_size)
            positives = ERP-nearest neighbors
            negatives = remaining items in group
        """
        categories, colors, fabrics, stores, temporal_features, images = batch
        batch_size = temporal_features.shape[0]
        num_group = batch_size // self.group_size

        # Compute embeddings
        emb = self.forward(categories, colors, fabrics, stores, temporal_features, images)

        # Pairwise Euclidean distance
        pairwise_dist = torch.cdist(emb, emb, p=2)

        # Anchor indices: first element of each group
        indices = torch.tensor([i * self.group_size for i in range(num_group)]).to('cuda')

        # Distance matrix for anchors to all others in their respective group
        anchor_ref_dist = torch.index_select(pairwise_dist, 0, indices)

        # Hardest positive = farthest among top-n neighbors
        hardest_positive_dist = torch.stack([
            torch.max(anchor_ref_dist[i,
                    1 + i * self.group_size : 1 + self.n_neighbors + i * self.group_size], 0)[0]
            for i in range(num_group)
        ])

        # Hardest negative = closest among the remaining items
        hardest_negative_dist = torch.stack([
            torch.min(anchor_ref_dist[i,
                    1 + self.n_neighbors + i * self.group_size : (i + 1) * self.group_size], 0)[0]
            for i in range(num_group)
        ])

        # Margin-based contrastive loss
        if self.margin_type == 'maxplus':
            triplet_loss = hardest_positive_dist - hardest_negative_dist + 1
            triplet_loss[triplet_loss < 0] = 0  # hinge
        elif self.margin_type == 'softplus':
            triplet_loss = torch.log(torch.exp(hardest_positive_dist - hardest_negative_dist) + 1)

        triplet_loss = triplet_loss.mean()

        # Logging
        self.log(f'{phase}_loss', triplet_loss)
        self.log(f'{phase}_hardest_positive_dist', hardest_positive_dist.mean())
        self.log(f'{phase}_hardest_negative_dist', hardest_negative_dist.mean())

        return triplet_loss

    # Evaluation version (same logic but separated for clarity)
    def phase_step_eval(self, batch, phase):
        return self.phase_step(batch, phase)

    # Prediction phase → return embeddings only
    def phase_step_pred(self, batch, phase):
        categories, colors, fabrics, stores, temporal_features, images = batch
        emb = self.forward(categories, colors, fabrics, stores, temporal_features, images)
        return emb


# -------------------------------------------------------------------------
# Full model (with forward pass)
# -------------------------------------------------------------------------
class ERP_neighbor_contrastive_learning(ERP_neighbor_contrastive_learning_predictor):
    """
    Actual model with forward() definition.
    Combines:
        - temporal encoder
        - attribute encoder
        - text encoder
        - image embedding
    Then fuses everything into a single item embedding.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, categories, colors, fabrics, stores, temporal_features, images):
        temp = self.temp_encoder(temporal_features)
        meta_data = self.attribute_encoder(categories, colors, fabrics, stores)
        texts = self.text_encoder(categories, colors, fabrics)

        item_embedding = self.feature_fusion_network(images, texts, temp, meta_data).unsqueeze(1)

        return item_embedding.squeeze()
