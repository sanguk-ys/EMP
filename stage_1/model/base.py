import torch
import pytorch_lightning as pl
from torchmetrics.regression import R2Score, SymmetricMeanAbsolutePercentageError

import urllib3
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.manifold import TSNE
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


class PytorchLightningBase(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.save_hyperparameters()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def training_step(self, train_batch, batch_idx):
        output = self.phase_step(train_batch, phase='train')
        return output

    def validation_step(self, valid_batch, batch_idx):
        self.eval()
        with torch.no_grad():
            self.phase_step_eval(valid_batch, phase='valid')

    def test_step(self, test_batch, batch_idx):
        self.eval()
        with torch.no_grad():
            self.phase_step_eval(test_batch, phase='test')

    def predict_step(self, test_batch, batch_idx):
        self.eval()
        with torch.no_grad():
            prediction = self.phase_step_pred(test_batch, phase='pred')
        return prediction

    def get_score(self, gt, pred):
        ad_smape = SymmetricMeanAbsolutePercentageError()
        r2 = R2Score()
        pred = pred.detach().cpu()
        gt = gt.detach().cpu()

        if gt.dim() == 1:
            adjust_smape = ad_smape(pred, gt) * 0.5
            r2_score = r2(pred, gt)
        else:
            adjust_smape = [ad_smape(pred[i], gt[i]) * 0.5 for i in range(len(gt))]
            adjust_smape = torch.mean(torch.stack(adjust_smape))

            r2_score = [r2(pred[i], gt[i]) for i in range(len(gt))]
            r2_score = torch.mean(torch.stack(r2_score))

        return adjust_smape, r2_score

    def emb_tsne_plotting(self, phase, embeddings):
        tsne_df = pd.DataFrame(TSNE(n_components=2).fit_transform(embeddings), columns=['component_0', 'component_1'])
        plt.scatter(tsne_df.iloc[0]['component_0'], tsne_df.iloc[0]['component_1'], color='red', label='anchor', zorder=3, s=100)
        plt.scatter(tsne_df.iloc[1:11]['component_0'], tsne_df.iloc[1:11]['component_1'], color='blue', label='pos', zorder=2)
        plt.scatter(tsne_df.iloc[11:]['component_0'], tsne_df.iloc[11:]['component_1'], color='yellow', label='neg', zorder=1)
        plt.legend()
        self.logger.log_image(key=f'{phase}', images=[plt])
        plt.clf()