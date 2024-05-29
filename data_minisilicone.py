import datasets
import gzip
import transformers

if __name__ == "__main__":
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        "google-t5/t5-small", cache_dir="./hf/tokenizers",
        model_max_length=128
    )
    #tokenizer = transformers.AutoTokenizer.from_pretrained(
    #    "facebook/bart-large", cache_dir="./hf/tokenizers",
    #    model_max_length=128
    #)

    def f(data_point, indices=None):
        top_tokens = tokenizer.tokenize(data_point["document"])[:128]
        top_tokens_ids = tokenizer.convert_tokens_to_ids(top_tokens)
        document = tokenizer.decode(top_tokens_ids, skip_special_tokens=False)
        bin_document = document.encode("utf-8")
        gzip_document = str(gzip.compress(bin_document, compresslevel=9))
        gzip_tokens = tokenizer.tokenize(gzip_document)
        return len(gzip_tokens) <= 128

    train_ds = datasets.load_dataset(
        "silicone", name="dyda_da", split=datasets.Split.TRAIN,
        cache_dir="./hf/datasets", trust_remote_code=True
    ).select_columns("Utterance")
    val_ds = datasets.load_dataset(
        "silicone", name="dyda_da", split=datasets.Split.VALIDATION,
        cache_dir="./hf/datasets", trust_remote_code=True
    ).select_columns("Utterance")
    test_ds = datasets.load_dataset(
        "silicone", name="dyda_da", split=datasets.Split.TEST,
        cache_dir="./hf/datasets", trust_remote_code=True
    ).select_columns("Utterance")

    train_ds = train_ds.rename_column("Utterance", "document")
    train_ds = train_ds.filter(function=f, batched=False)
    train_ds.save_to_disk(
        "./hf/datasets/t5-small/minisilicone/train", max_shard_size="2MB"
    )
    #train_ds.save_to_disk(
    #    "./hf/datasets/bart-large/minisilicone/train", max_shard_size="2MB"
    #)

    val_ds = val_ds.rename_column("Utterance", "document")
    val_ds = val_ds.filter(function=f, batched=False)
    val_ds.save_to_disk(
        "./hf/datasets/t5-small/minisilicone/validation", max_shard_size="2MB"
    )
    #val_ds.save_to_disk(
    #    "./hf/datasets/bart-large/minisilicone/validation", max_shard_size="2MB"
    #)

    test_ds = test_ds.rename_column("Utterance", "document")
    test_ds = test_ds.filter(function=f, batched=False)
    test_ds.save_to_disk(
        "./hf/datasets/t5-small/minisilicone/test", max_shard_size="2MB"
    )
    #test_ds.save_to_disk(
    #    "./hf/datasets/bart-large/minisilicone/test", max_shard_size="2MB"
    #)