import pytorch_lightning
import pytorch_lightning.callbacks
import pytorch_lightning.loggers

SEED = 0xdeadbeef

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
GPU_TRAINER = pytorch_lightning.Trainer(
    accelerator="gpu",
    logger=logger,
    callbacks=callbacks,
    min_epochs=1,
    max_epochs=-1,
    enable_checkpointing=True,
    enable_progress_bar=True,
    accumulate_grad_batches=1,
    gradient_clip_val=0.1,
    gradient_clip_algorithm="norm",
    precision="16-mixed",
    log_every_n_steps=25
)
CPU_TRAINER = pytorch_lightning.Trainer(
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

GPU_TEST_TRAINER = pytorch_lightning.Trainer(
    accelerator="gpu",
    logger=None,
    callbacks=[pytorch_lightning.callbacks.RichProgressBar(leave=False)],
    min_epochs=1,
    max_epochs=-1,
    enable_checkpointing=False,
    enable_model_summary=False,
    enable_progress_bar=True,
)
CPU_TEST_TRAINER = pytorch_lightning.Trainer(
    accelerator="cpu",
    logger=None,
    callbacks=[pytorch_lightning.callbacks.RichProgressBar(leave=False)],
    min_epochs=1,
    max_epochs=-1,
    enable_checkpointing=False,
    enable_model_summary=False,
    enable_progress_bar=True,
)
