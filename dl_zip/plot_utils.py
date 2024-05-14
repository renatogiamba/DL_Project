import polars
import plotly
import plotly.express
import plotly.graph_objects
from typing import *

def compute_agg_metrics(metric_files: List[str]) -> polars.DataFrame:
    df = polars.concat(
        map(
            lambda metric_file: \
            polars.scan_csv(metric_file, has_header=True, separator=","),
            metric_files
        ), how="vertical"
    )
    
    df_train = df.select(
        polars.col("epoch"),
        polars.col("train_loss_epoch"),
        polars.col("train_dist_epoch")
    )
    df_train = df_train.cast({
        "epoch": polars.UInt32,
        "train_loss_epoch": polars.Float32,
        "train_dist_epoch": polars.Float32
    })
    df_train = df_train.filter(polars.col("epoch") <= 10)
    df_train = df_train.drop_nulls(subset="epoch")
    df_train = df_train.drop_nulls(subset="train_loss_epoch")
    df_train = df_train.drop_nulls(subset="train_dist_epoch")

    df_validation = df.select(
        polars.col("epoch"),
        polars.col("val_loss_epoch"),
        polars.col("val_dist_epoch")
    )
    df_validation = df_validation.cast({
        "epoch": polars.UInt32,
        "val_loss_epoch": polars.Float32,
        "val_dist_epoch": polars.Float32
    })
    df_validation = df_validation.filter(polars.col("epoch") <= 10)
    df_validation = df_validation.drop_nulls(subset="epoch")
    df_validation = df_validation.drop_nulls(subset="val_loss_epoch")
    df_validation = df_validation.drop_nulls(subset="val_dist_epoch")
    
    df = df_train.join(df_validation, on="epoch", how="inner", validate="1:1")
    return df.collect()

def load_metrics(metric_file: str) -> polars.DataFrame:
    return polars.scan_parquet(metric_file).collect()

def save_metrics(metric_df: polars.DataFrame, metric_file: str) -> None:
    metric_df.write_parquet(metric_file)
