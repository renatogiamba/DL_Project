import dl_zip

if __name__ == "__main__":
    df = dl_zip.compute_agg_metrics([
        f"results/t5-small/MiniSilicone/metrics({i}).csv" for i in range(1, 7)
    ])
    print(df)
    dl_zip.save_metrics(df, "results/t5-small/MiniSilicone/metrics_agg.parquet")
    df = dl_zip.load_metrics("results/t5-small/MiniSilicone/metrics_agg.parquet")
    print(df)

    df = dl_zip.compute_agg_metrics([
        f"results/t5-small/Silicone/metrics({i}).csv" for i in range(1, 10)
    ])
    print(df)
    dl_zip.save_metrics(df, "results/t5-small/Silicone/metrics_agg.parquet")
    df = dl_zip.load_metrics("results/t5-small/Silicone/metrics_agg.parquet")
    print(df)

    df = dl_zip.compute_agg_metrics([
        f"results/t5-small/StanfordSST2/metrics(1).csv"
    ])
    print(df)
    dl_zip.save_metrics(df, "results/t5-small/StanfordSST2/metrics_agg.parquet")
    df = dl_zip.load_metrics("results/t5-small/StanfordSST2/metrics_agg.parquet")
    print(df)
