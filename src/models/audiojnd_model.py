import pytorch_lightning as pl
import torch.nn as nn
import torch
from sklearn.metrics import roc_auc_score

from src.utils.technical_utils import load_obj
from torchopenl3.models import load_audio_embedding_model


class ScaleLayer(nn.Module):
    def __init__(self, init_value=1e-3):
        super().__init__()
        self.scale = nn.Parameter(torch.FloatTensor([init_value] * 6144))

    def forward(self, input):
        return input * self.scale


class AudioJNDModel(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.model = load_audio_embedding_model(
            input_repr=self.cfg.model.input_repr,
            content_type=self.cfg.model.content_type,
            embedding_size=self.cfg.model.embedding_size,
        )
        self.scale = ScaleLayer()
        # Could also try l1 with crossentropy
        self.cos = nn.CosineSimilarity(eps=1e-6)
        self.criterion = load_obj(self.cfg.loss.class_name)()

    def forward(self, x1, x2):
        bs, _, in2, in3 = x1.size()

        x1 = self.model(x1.view(-1, in2, in3))
        x2 = self.model(x2.view(-1, in2, in3))

        x1 = self.scale(x1).view(bs, -1)
        x2 = self.scale(x2).view(bs, -1)

        prob = self.cos(x1, x2)
        assert torch.all((prob < 1) & (prob > -1))

        prob = (prob + 1) / 2
        assert torch.all((prob < 1) & (prob > 0))

        return prob

    def training_step(self, batch, batch_idx):

        x1, x2, labels = batch
        labels = labels.float()
        output = self.forward(x1, x2)

        loss = self.criterion(output, labels)

        try:
            auc = roc_auc_score(labels.detach().cpu(), output.detach().cpu())
            self.log("auc", auc, on_step=True, prog_bar=True, logger=True)
            self.log("Train Loss", loss, on_step=True, prog_bar=True, logger=True)
        except:
            pass
        return {"loss": loss, "preds": output, "targets": labels}

    def training_epoch_end(self, outputs):

        preds = []
        labels = []

        for output in outputs:

            preds += output["preds"]
            labels += output["targets"]

        labels = torch.stack(labels)
        preds = torch.stack(preds)

        train_auc = roc_auc_score(labels.detach().cpu(), preds.detach().cpu())
        self.log("train_auc", train_auc, prog_bar=True, logger=True)

    def validation_step(self, batch, batch_idx):
        x1, x2, labels = batch
        labels = labels.float()
        output = self.forward(x1, x2)
        loss = self.criterion(output, labels)

        self.log("val_loss", loss, on_step=True, prog_bar=True, logger=True)
        return {"preds": output, "targets": labels}

    def validation_epoch_end(self, outputs):

        preds = []
        labels = []

        for output in outputs:
            preds += output["preds"]
            labels += output["targets"]

        labels = torch.stack(labels)
        preds = torch.stack(preds)

        val_auc = roc_auc_score(labels.detach().cpu(), preds.detach().cpu())
        self.log("val_auc", val_auc, prog_bar=True, logger=True)

    def configure_optimizers(self):
        optimizer = load_obj(self.cfg.optimizer.class_name)(
            self.model.parameters(), **self.cfg.optimizer.params
        )
        scheduler = load_obj(self.cfg.scheduler.class_name)(optimizer, **self.cfg.scheduler.params)

        return (
            [optimizer],
            [
                {
                    "scheduler": scheduler,
                    "interval": self.cfg.scheduler.step,
                    "monitor": self.cfg.scheduler.monitor,
                }
            ],
        )
