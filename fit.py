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

    callbacks = [
        pytorch_lightning.callbacks.RichModelSummary(max_depth=-1),
        pytorch_lightning.callbacks.RichProgressBar(leave=False),
        pytorch_lightning.callbacks.EarlyStopping(
            monitor="val_loss_epoch", min_delta=0, patience=5, mode="min",
            strict=True, check_on_train_epoch_end=True
        ),
        pytorch_lightning.callbacks.ModelCheckpoint(
            dirpath="./model", filename="{epoch:03d}--{val_loss_epoch:.4f}",
            monitor="val_loss_epoch", mode="min",
            save_on_train_epoch_end=True
        )
    ]
    logger = pytorch_lightning.loggers.CSVLogger("./lightning", name="logs")
    #trainer = pytorch_lightning.Trainer(
    #    accelerator="gpu",
    #    logger=logger,
    #    callbacks=callbacks,
    #    min_epochs=1,
    #    max_epochs=-1,
    #    enable_checkpointing=True,
    #    enable_progress_bar=True,
    #    accumulate_grad_batches=1,
    #    gradient_clip_val=0.1,
    #    gradient_clip_algorithm="norm",
    #    precision="16-mixed",
    #    log_every_n_steps=25
    #)
    trainer = pytorch_lightning.Trainer(
    accelerator="cpu",
    logger=logger,
    callbacks=callbacks,
    min_epochs=1,
    max_epochs=-1,
    enable_checkpointing=True,
    enable_progress_bar=True,
    accumulate_grad_batches=1,
    gradient_clip_val=0.1,
    gradient_clip_algorithm="norm",
    log_every_n_steps=25
)

    #trainer.fit(model, datamodule=dm)
    #trainer.fit(model, datamodule=dm, ckpt_path="{}.ckpt")
