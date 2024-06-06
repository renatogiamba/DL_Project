import dl_zip

if __name__ == "__main__":
    t5_agg_metrics_file = "results/t5-small/{}/metrics_agg.parquet"
    t5_metrics_file = "results/t5-small/{}/metrics.parquet"
    t5_csv_metrics_file = "results/t5-small/{}/metrics({}).csv"
    bart_agg_metrics_file = "results/bart-large/{}/metrics_agg.parquet"
    bart_metrics_file = "results/bart-large/{}/metrics.parquet"
    bart_csv_metrics_file = "results/bart-large/{}/metrics({}).csv"

    metrics_df = dl_zip.compute_metrics([
        t5_csv_metrics_file.format("MiniSilicone", i) for i in range(1, 3)
    ], 9)
    dl_zip.save_metrics(metrics_df, t5_metrics_file.format("MiniSilicone"))
    metrics_df = dl_zip.compute_agg_metrics([
        t5_csv_metrics_file.format("MiniSilicone", i) for i in range(1, 3)
    ], 9)
    dl_zip.save_metrics(metrics_df, t5_agg_metrics_file.format("MiniSilicone"))

    metrics_df = dl_zip.compute_metrics([
        t5_csv_metrics_file.format("Silicone", i) for i in range(1, 10)
    ], 7)
    dl_zip.save_metrics(metrics_df, t5_metrics_file.format("Silicone"))
    metrics_df = dl_zip.compute_agg_metrics([
        t5_csv_metrics_file.format("Silicone", i) for i in range(1, 10)
    ], 7)
    dl_zip.save_metrics(metrics_df, t5_agg_metrics_file.format("Silicone"))

    metrics_df = dl_zip.compute_metrics([
        t5_csv_metrics_file.format("StanfordSST2", i) for i in range(1, 2)
    ], 5)
    dl_zip.save_metrics(metrics_df, t5_metrics_file.format("StanfordSST2"))
    metrics_df = dl_zip.compute_agg_metrics([
        t5_csv_metrics_file.format("StanfordSST2", i) for i in range(1, 2)
    ], 5)
    dl_zip.save_metrics(metrics_df, t5_agg_metrics_file.format("StanfordSST2"))

    metrics_df = dl_zip.compute_metrics([
        bart_csv_metrics_file.format("MiniSilicone", i) for i in range(1, 2)
    ], 3)
    dl_zip.save_metrics(metrics_df, bart_metrics_file.format("MiniSilicone"))
    metrics_df = dl_zip.compute_agg_metrics([
        bart_csv_metrics_file.format("MiniSilicone", i) for i in range(1, 2)
    ], 3)
    dl_zip.save_metrics(metrics_df, bart_agg_metrics_file.format("MiniSilicone"))

    metrics_df = dl_zip.compute_metrics([
        bart_csv_metrics_file.format("StanfordSST2", i) for i in range(1, 2)
    ], 2)
    dl_zip.save_metrics(metrics_df, bart_metrics_file.format("StanfordSST2"))
    metrics_df = dl_zip.compute_agg_metrics([
        bart_csv_metrics_file.format("StanfordSST2", i) for i in range(1, 2)
    ], 2)
    dl_zip.save_metrics(metrics_df, bart_agg_metrics_file.format("StanfordSST2"))
