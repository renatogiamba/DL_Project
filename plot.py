import dl_zip

if __name__ == "__main__":
    t5_agg_metrics_file = "results/t5-small/{}/metrics_agg.parquet"
    t5_agg_loss_img = "results/t5-small/{}/loss_agg.png"
    t5_agg_dist_img = "results/t5-small/{}/dist_agg.png"
    t5_metrics_file = "results/t5-small/{}/metrics.parquet"
    t5_loss_img = "results/t5-small/{}/loss.png"
    t5_dist_img = "results/t5-small/{}/dist.png"
    bart_agg_metrics_file = "results/bart-large/{}/metrics_agg.parquet"
    bart_agg_loss_img = "results/bart-large/{}/loss_agg.png"
    bart_agg_dist_img = "results/bart-large/{}/dist_agg.png"
    bart_metrics_file = "results/bart-large/{}/metrics.parquet"
    bart_loss_img = "results/bart-large/{}/loss.png"
    bart_dist_img = "results/bart-large/{}/dist.png"

    df_loss, df_dist = dl_zip.metrics_for_plot(
        t5_metrics_file.format("MiniSilicone"), "step"
    )
    dl_zip.plot(df_loss, "step", "loss")\
        .write_image(t5_loss_img.format("MiniSilicone"))
    dl_zip.plot(df_dist, "step", "dist")\
        .write_image(t5_dist_img.format("MiniSilicone"))
    
    df_loss, df_dist = dl_zip.metrics_for_plot(
        t5_agg_metrics_file.format("MiniSilicone"), "epoch"
    )
    dl_zip.plot(df_loss, "epoch", "loss")\
        .write_image(t5_agg_loss_img.format("MiniSilicone"))
    dl_zip.plot(df_dist, "epoch", "dist")\
        .write_image(t5_agg_dist_img.format("MiniSilicone"))
    
    df_loss, df_dist = dl_zip.metrics_for_plot(
        t5_metrics_file.format("Silicone"), "step"
    )
    dl_zip.plot(df_loss, "step", "loss")\
        .write_image(t5_loss_img.format("Silicone"))
    dl_zip.plot(df_dist, "step", "dist")\
        .write_image(t5_dist_img.format("Silicone"))
    
    df_loss, df_dist = dl_zip.metrics_for_plot(
        t5_agg_metrics_file.format("Silicone"), "epoch"
    )
    dl_zip.plot(df_loss, "epoch", "loss")\
        .write_image(t5_agg_loss_img.format("Silicone"))
    dl_zip.plot(df_dist, "epoch", "dist")\
        .write_image(t5_agg_dist_img.format("Silicone"))
    
    df_loss, df_dist = dl_zip.metrics_for_plot(
        t5_metrics_file.format("StanfordSST2"), "step"
    )
    dl_zip.plot(df_loss, "step", "loss")\
        .write_image(t5_loss_img.format("StanfordSST2"))
    dl_zip.plot(df_dist, "step", "dist")\
        .write_image(t5_dist_img.format("StanfordSST2"))
    
    df_loss, df_dist = dl_zip.metrics_for_plot(
        t5_agg_metrics_file.format("StanfordSST2"), "epoch"
    )
    dl_zip.plot(df_loss, "epoch", "loss")\
        .write_image(t5_agg_loss_img.format("StanfordSST2"))
    dl_zip.plot(df_dist, "epoch", "dist")\
        .write_image(t5_agg_dist_img.format("StanfordSST2"))
    
    df_loss, df_dist = dl_zip.metrics_for_plot(
        bart_metrics_file.format("MiniSilicone"), "step"
    )
    dl_zip.plot(df_loss, "step", "loss")\
        .write_image(bart_loss_img.format("MiniSilicone"))
    dl_zip.plot(df_dist, "step", "dist")\
        .write_image(bart_dist_img.format("MiniSilicone"))
    
    df_loss, df_dist = dl_zip.metrics_for_plot(
        bart_agg_metrics_file.format("MiniSilicone"), "epoch"
    )
    dl_zip.plot(df_loss, "epoch", "loss")\
        .write_image(bart_agg_loss_img.format("MiniSilicone"))
    dl_zip.plot(df_dist, "epoch", "dist")\
        .write_image(bart_agg_dist_img.format("MiniSilicone"))
    
    df_loss, df_dist = dl_zip.metrics_for_plot(
        bart_metrics_file.format("StanfordSST2"), "step"
    )
    dl_zip.plot(df_loss, "step", "loss")\
        .write_image(bart_loss_img.format("StanfordSST2"))
    dl_zip.plot(df_dist, "step", "dist")\
        .write_image(bart_dist_img.format("StanfordSST2"))
    
    df_loss, df_dist = dl_zip.metrics_for_plot(
        bart_agg_metrics_file.format("StanfordSST2"), "epoch"
    )
    dl_zip.plot(df_loss, "epoch", "loss")\
        .write_image(bart_agg_loss_img.format("StanfordSST2"))
    dl_zip.plot(df_dist, "epoch", "dist")\
        .write_image(bart_agg_dist_img.format("StanfordSST2"))
