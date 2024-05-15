import polars
import plotly
import plotly.express
import plotly.graph_objects

if __name__ == "__main__":
    df = polars.scan_csv(
        "results/t5-small/StanfordSST2/metrics(1).csv",
        has_header=True, separator=","
    )
    df = df.select(
        polars.col("epoch"),
        polars.col("step"),
        polars.col("train_loss_step"),
        polars.col("train_dist_step"),
        polars.col("val_loss_step"),
        polars.col("val_dist_step")
    )
    df_train = df.select(
        polars.col("epoch"),
        polars.col("step"),
        polars.col("train_loss_step"),
        polars.col("train_dist_step")
    )
    df_train = df_train.drop_nulls(subset="epoch")
    df_train = df_train.drop_nulls(subset="train_loss_step")
    df_train = df_train.drop_nulls(subset="train_dist_step")
    df_train = df_train.drop("epoch")
    df_validation = df.select(
        polars.col("epoch"),
        polars.col("step"),
        polars.col("val_loss_step"),
        polars.col("val_dist_step")
    )
    df_validation = df_validation.drop_nulls(subset="val_loss_step")
    df_validation = df_validation.drop_nulls(subset="val_dist_step")
    df_validation = df_validation.drop("epoch")
    df = df_train.join(
        df_validation, on="step", how="outer",
        validate="m:m")
    df1 = df.drop_nulls(subset="step").drop("step_right")
    df2 = df.drop_nulls(subset="step_right").drop("step")
    df2 = df2.select(
        polars.col("step_right").alias("step"),
        polars.col("train_loss_step"),
        polars.col("train_dist_step"),
        polars.col("val_loss_step"),
        polars.col("val_dist_step")
    )
    df = polars.concat([df1, df2], how="vertical").sort("step")
    df_train.collect().write_csv(
        "results/t5-small/StanfordSST2/metrics_train.csv",
        include_header=True, separator=";"
    )
    df_validation.collect().write_csv(
        "results/t5-small/StanfordSST2/metrics_validation.csv",
        include_header=True, separator=";"
    )
    df.collect().write_csv(
        "results/t5-small/StanfordSST2/metrics.csv",
        include_header=True, separator=";"
    )