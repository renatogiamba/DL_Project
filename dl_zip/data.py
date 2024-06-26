import datasets
import gzip
import pytorch_lightning
import struct
import torch
import torch.nn
import torch.nn.utils
import torch.nn.utils.rnn
import torch.utils
import torch.utils.data
import transformers
import zlib
from typing import *

class ZipDataCollator():
    def __init__(self, tokenizer: transformers.PreTrainedTokenizer):
        self.tokenizer = tokenizer

    def __call__(
            self,
            data_points: List[Dict[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        input_ids = list()
        attention_mask = list()
        labels = list()
        gzip_headers = list()
        gzip_footers = list()

        for data_point in data_points:
            input_ids.append(data_point["input_ids"])
            attention_mask.append(data_point["attention_mask"])
            labels.append(data_point["labels"])
            gzip_headers.append(data_point["gzip_header"])
            gzip_footers.append(data_point["gzip_footer"])

        return {
            "input_ids": torch.nn.utils.rnn.pad_sequence(
                input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
            ).to(dtype=torch.int64),
            "attention_mask": torch.nn.utils.rnn.pad_sequence(
                attention_mask, batch_first=True, padding_value=0.
            ).to(dtype=torch.int64),
            "labels": torch.nn.utils.rnn.pad_sequence(
                labels, batch_first=True, padding_value=-100.
            ).to(dtype=torch.int64),
            "gzip_headers": torch.stack(gzip_headers, dim=0),
            "gzip_footers": torch.stack(gzip_footers, dim=0)
        }

class ZipDataModule(pytorch_lightning.LightningDataModule):
    def __init__(
            self,
            hf_model_name: str,
            batch_size: int,
            max_seq_len: int = 128
    ) -> None:
        super(ZipDataModule, self).__init__()

        self.hf_model_name = hf_model_name
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.tokenizer = None
        self.train_ds = None
        self.val_ds = None
        self.test_ds = None

    def prepare_data(self) -> None:
        transformers.AutoTokenizer.from_pretrained(
            self.hf_model_name,
            cache_dir="./hf/tokenizers",
            model_max_length=self.max_seq_len
        )

    def setup(self, stage: str) -> None:
        def gzip_file(data_point, indices=None):
            top_tokens = self.tokenizer.tokenize(data_point["document"])[:self.max_seq_len]
            top_tokens_ids = self.tokenizer.convert_tokens_to_ids(top_tokens)
            document = self.tokenizer.decode(top_tokens_ids, skip_special_tokens=False)
            bin_document = document.encode("utf-8")
            return {
                "gzip_header": list(
                    b"\x1f\x8b\x08\x00" + struct.pack("<I", 0) + b"\x02\xff"
                ),
                "gzip_footer": list(
                    struct.pack("<I", zlib.crc32(bin_document)) \
                        + struct.pack("<I", len(bin_document))
                ),
            }

        def preprocess(batch, indices=None):
            features = self.tokenizer(
                text=batch["document"],
                padding=False,
                truncation=True,
                is_split_into_words=False,
                return_attention_mask=True
            )
            labels = self.tokenizer(
                text=[
                    gzip.compress(doc.encode("utf-8"), compresslevel=9)[10:-8].hex()
                    for doc in batch["document"]
                ],
                padding=False,
                truncation=True,
                is_split_into_words=False,
                return_attention_mask=False
            )
            return {
                "input_ids": features["input_ids"],
                "attention_mask": features["attention_mask"],
                "labels": labels["input_ids"]
            }

        if stage == "fit":
            self.tokenizer = transformers.AutoTokenizer.from_pretrained(
                self.hf_model_name,
                cache_dir="./hf/tokenizers",
                model_max_length=self.max_seq_len
            )

            self.train_ds = self.train_ds.map(function=gzip_file, batched=False)
            self.train_ds = self.train_ds.map(function=preprocess, batched=True)
            self.train_ds.set_format(
                type="pt",
                columns=[
                    "input_ids", "attention_mask", "labels",
                    "gzip_header", "gzip_footer"
                ]
            )

            self.val_ds = self.val_ds.map(function=gzip_file, batched=False)
            self.val_ds = self.val_ds.map(function=preprocess, batched=True)
            self.val_ds.set_format(
                type="pt",
                columns=[
                    "input_ids", "attention_mask", "labels",
                    "gzip_header", "gzip_footer"
                ]
            )
        elif stage == "validate":
            self.tokenizer = transformers.AutoTokenizer.from_pretrained(
                self.hf_model_name,
                cache_dir="./hf/tokenizers",
                model_max_length=self.max_seq_len
            )

            self.val_ds = self.val_ds.map(function=gzip_file, batched=False)
            self.val_ds = self.val_ds.map(function=preprocess, batched=True)
            self.val_ds.set_format(
                type="pt",
                columns=[
                    "input_ids", "attention_mask", "labels",
                    "gzip_header", "gzip_footer"
                ]
            )
        elif stage == "test":
            self.tokenizer = transformers.AutoTokenizer.from_pretrained(
                self.hf_model_name,
                cache_dir="./hf/tokenizers",
                model_max_length=self.max_seq_len
            )

            self.test_ds = self.test_ds.map(function=gzip_file, batched=False)
            self.test_ds = self.test_ds.map(function=preprocess, batched=True)
            self.test_ds.set_format(
                type="pt",
                columns=[
                    "input_ids", "attention_mask", "labels",
                    "gzip_header", "gzip_footer"
                ]
            )

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(
            self.train_ds, batch_size=self.batch_size, shuffle=True,
            collate_fn=ZipDataCollator(self.tokenizer)
        )

    def val_dataloader(self) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(
            self.val_ds, batch_size=self.batch_size, shuffle=False,
            collate_fn=ZipDataCollator(self.tokenizer)
        )

    def test_dataloader(self) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(
            self.test_ds, batch_size=1, shuffle=False,
            collate_fn=ZipDataCollator(self.tokenizer)
        )

class MiniSiliconeDataModule(ZipDataModule):
    def __init__(
            self,
            hf_model_name: str,
            batch_size: int,
            max_seq_len: int = 128
    ) -> None:
        super(MiniSiliconeDataModule, self).__init__(
            hf_model_name,
            batch_size,
            max_seq_len=max_seq_len
        )

    def prepare_data(self) -> None:
        super(MiniSiliconeDataModule, self).prepare_data()

    def setup(self, stage: str) -> None:
        if stage == "fit":
            self.train_ds = datasets.load_from_disk(
                "./hf/datasets/filtered/minisilicone/train"
            )

            self.val_ds = datasets.load_from_disk(
                "./hf/datasets/filtered/minisilicone/validation"
            )

        elif stage == "validate":
            self.val_ds = datasets.load_from_disk(
                "./hf/datasets/filtered/minisilicone/validation"
            )

        elif stage == "test":
            self.test_ds = datasets.load_from_disk(
                "./hf/datasets/filtered/minisilicone/test"
            )

        super(MiniSiliconeDataModule, self).setup(stage)
    
class SiliconeDataModule(ZipDataModule):
    def __init__(
            self,
            hf_model_name: str,
            batch_size: int,
            max_seq_len: int = 128
    ) -> None:
        super(SiliconeDataModule, self).__init__(
            hf_model_name,
            batch_size,
            max_seq_len=max_seq_len
        )

    def prepare_data(self) -> None:
        super(SiliconeDataModule, self).prepare_data()

    def setup(self, stage: str) -> None:
        if stage == "fit":
            self.train_ds = datasets.load_from_disk(
                "./hf/datasets/filtered/silicone/train"
            )

            self.val_ds = datasets.load_from_disk(
                "./hf/datasets/filtered/silicone/validation"
            )

        elif stage == "validate":
            self.val_ds = datasets.load_from_disk(
                "./hf/datasets/filtered/silicone/validation"
            )

        elif stage == "test":
            self.test_ds = datasets.load_from_disk(
                "./hf/datasets/filtered/silicone/test"
            )

        super(SiliconeDataModule, self).setup(stage)

class StanfordSST2DataModule(ZipDataModule):
    def __init__(
            self,
            hf_model_name: str,
            batch_size: int,
            max_seq_len: int = 128
    ) -> None:
        super(StanfordSST2DataModule, self).__init__(
            hf_model_name,
            batch_size,
            max_seq_len=max_seq_len
        )

    def prepare_data(self) -> None:
        super(StanfordSST2DataModule, self).prepare_data()

    def setup(self, stage: str) -> None:
        if stage == "fit":
            self.train_ds = datasets.load_from_disk(
                "./hf/datasets/filtered/stanford_sst2/train"
            )

            self.val_ds = datasets.load_from_disk(
                "./hf/datasets/filtered/stanford_sst2/validation"
            )

        elif stage == "validate":
            self.val_ds = datasets.load_from_disk(
                "./hf/datasets/filtered/stanford_sst2/validation"
            )

        elif stage == "test":
            self.test_ds = datasets.load_from_disk(
                "./hf/datasets/filtered/stanford_sst2/test"
            )

        super(StanfordSST2DataModule, self).setup(stage)
