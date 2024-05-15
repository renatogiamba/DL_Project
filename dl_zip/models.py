import pytorch_lightning
import torch
import torch.nn
import torch.nn.functional
import torch.optim
import torch.optim.lr_scheduler
import torchmetrics
import torchmetrics.functional
import torchmetrics.functional.text
import transformers
from typing import *

class ZipModel(pytorch_lightning.LightningModule):
    def __init__(
            self,
            hf_model_name: str,
            lr: float = 1.e-4
    ) -> None:
        super(ZipModel, self).__init__()

        self.save_hyperparameters()

        self.transf_model = transformers.AutoModelForSeq2SeqLM.from_pretrained(
            hf_model_name,
            cache_dir="./hf/models"
        )
        self.manual_freeze()

    def manual_freeze(self) -> None:
        pass

    def forward(
            self,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor,
            labels: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        transf_outputs = self.transf_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        return transf_outputs.loss, transf_outputs.logits

    def levenshtein_edit_distance_fn(
            self,
            pred_zipped_bytes_seqs_logits: torch.Tensor,
            zipped_bytes_seqs: torch.Tensor
    ) -> torch.Tensor:
        pred_zipped_bytes_seqs = torch.argmax(pred_zipped_bytes_seqs_logits[:, 1:, :], dim=2).tolist()
        zipped_bytes_seqs = zipped_bytes_seqs[:, 1:].tolist()
        pred_zipped_chars = list(map(lambda byte_seq: "".join(list(map(lambda byte: chr(byte), byte_seq))), pred_zipped_bytes_seqs))
        zipped_chars = list(map(lambda byte_seq: "".join(list(map(lambda byte: chr(byte), byte_seq))), zipped_bytes_seqs))
        return torchmetrics.functional.text.edit_distance(pred_zipped_chars, zipped_chars)

    def our_distance_fn(
            self,
            pred_zipped_bytes_seqs_logits: torch.Tensor,
            zipped_bytes_seqs: torch.Tensor
    ) -> torch.Tensor:
        pred_zipped_bytes_seqs = torch.argmax(pred_zipped_bytes_seqs_logits, dim=2)
        mask = (pred_zipped_bytes_seqs != zipped_bytes_seqs)
        mask = torch.logical_and(mask, zipped_bytes_seqs != -100)
        sum_dist = torch.sum(mask.to(dtype=torch.uint8), dim=1).to(dtype=torch.float32)
        return torch.mean(sum_dist) / pred_zipped_bytes_seqs_logits.shape[1] * 100.

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.hparams.lr,
            betas=(0.9, 0.98), eps=1.e-9, weight_decay=0.01
        )
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.1, patience=2
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "monitor": "val_loss_epoch",
                "interval": "epoch",
                "frequency": 1
            }
        }

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        loss, pred_zipped_bytes_seqs_logits = self(
            batch["input_ids"],
            batch["attention_mask"],
            batch["labels"]
        )
        dist = self.our_distance_fn(
            pred_zipped_bytes_seqs_logits, batch["labels"]
        )

        self.log(
            "train_loss", loss,
            prog_bar=True, logger=True,
            on_step=True, on_epoch=True
        )
        self.log(
            "train_dist", dist,
            prog_bar=True, logger=True,
            on_step=True, on_epoch=True
        )

        return {
            "loss": loss,
            "dist": dist
        }

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        loss, pred_zipped_bytes_seqs_logits = self(
            batch["input_ids"],
            batch["attention_mask"],
            batch["labels"]
        )
        dist = self.our_distance_fn(
            pred_zipped_bytes_seqs_logits, batch["labels"]
        )

        self.log(
            "val_loss", loss,
            prog_bar=True, logger=True,
            on_step=True, on_epoch=True
        )
        self.log(
            "val_dist", dist,
            prog_bar=True, logger=True,
            on_step=True, on_epoch=True
        )

        return {
            "loss": loss,
            "dist": dist
        }

    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        loss, pred_zipped_bytes_seqs_logits = self(
            batch["input_ids"],
            batch["attention_mask"],
            batch["labels"]
        )
        dist = self.our_distance_fn(
            pred_zipped_bytes_seqs_logits, batch["labels"]
        )

        self.log(
            "test_loss", loss,
            prog_bar=True, logger=True,
            on_step=True, on_epoch=True
        )
        self.log(
            "test_dist", dist,
            prog_bar=True, logger=True,
            on_step=True, on_epoch=True
        )

        return {
            "loss": loss,
            "dist": dist
        }

class T5_Small_ZipModel(ZipModel):
    def __init__(self, lr: float = 1.e-4) -> None:
        super(T5_Small_ZipModel, self).__init__(
            "google-t5/t5-small",
            lr=lr
        )

    def manual_freeze(self) -> None:
        pass

class BART_Large_ZipModel(ZipModel):
    def __init__(self, lr: float = 1.e-4) -> None:
        super(BART_Large_ZipModel, self).__init__(
            "facebook/bart-large",
            lr=lr
        )

    def manual_freeze(self) -> None:
        self.transf_model.model.shared\
            .requires_grad_(requires_grad=False)
        for i in range(8):
            self.transf_model.model.encoder.layers[i]\
                .requires_grad_(requires_grad=False)
        for i in range(8):
            self.transf_model.model.decoder.layers[i]\
                .requires_grad_(requires_grad=False)
