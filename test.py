import dl_zip

if __name__ == "__main__":
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

    trainer = dl_zip.CPU_TEST_TRAINER
    #trainer = dl_zip.GPU_TEST_TRAINER

    trainer.test(model, datamodule=dm)
