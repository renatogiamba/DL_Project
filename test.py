import pytorch_lightning
import pytorch_lightning.callbacks
import pytorch_lightning.loggers

import dl_zip

if __name__ == "__main__":
    pytorch_lightning.seed_everything(dl_zip.SEED)
    
    HF_MODEL_NAME = "google-t5/t5-small"
    #HF_MODEL_NAME = "facebook/bart-large"

    #dm = dl_zip.MiniSiliconeDataModule(HF_MODEL_NAME, 64, max_seq_len=128)
    dm = dl_zip.MiniSiliconeDataModule(HF_MODEL_NAME, 32, max_seq_len=128)
    #dm = dl_zip.SiliconeDataModule(HF_MODEL_NAME, 64, max_seq_len=128)
    #dm = dl_zip.SiliconeDataModule(HF_MODEL_NAME, 32, max_seq_len=128)
    #dm = dl_zip.StanfordSST2DataModule(HF_MODEL_NAME, 64, max_seq_len=128)
    #dm = dl_zip.StanfordSST2DataModule(HF_MODEL_NAME, 32, max_seq_len=128)

    model = dl_zip.T5_Small_ZipModel(lr=1.e-4)
    #model = dl_zip.T5_Small_ZipModel.load_from_checkpoint(
    #    "{}.ckpt",
    #    hparams_file="{}hparams.yaml"
    #)
    #model = dl_zip.BART_Large_ZipModel(lr=1.e-4)
    #model = dl_zip.BART_Large_ZipModel.load_from_checkpoint(
    #    "{}.ckpt",
    #    hparams_file="{}hparams.yaml"
    #)

    #trainer = pytorch_lightning.Trainer(
    #    accelerator="gpu",
    #    logger=None,
    #    callbacks=[pytorch_lightning.callbacks.RichProgressBar(leave=False)],
    #    min_epochs=1,
    #    max_epochs=-1,
    #    enable_checkpointing=False,
    #    enable_model_summary=False,
    #    enable_progress_bar=True,
    #)
    trainer = pytorch_lightning.Trainer(
        accelerator="cpu",
        logger=None,
        callbacks=[pytorch_lightning.callbacks.RichProgressBar(leave=False)],
        min_epochs=1,
        max_epochs=-1,
        enable_checkpointing=False,
        enable_model_summary=False,
        enable_progress_bar=True,
    )

    trainer.test(model, datamodule=dm)
