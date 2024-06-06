import polars
import plotly
import plotly.express
import plotly.graph_objects
from typing import *

def compute_metrics(metric_files: List[str], last_epoch: int) -> polars.DataFrame:
    df = polars.concat(
        map(
            lambda metric_file: \
            polars.scan_csv(metric_file, has_header=True, separator=",")\
                .cast({
                    "epoch": polars.UInt32,
                    "step": polars.UInt64,
                    "train_loss_step": polars.Float32,
                    "train_loss_epoch": polars.Float32,
                    "train_dist_step": polars.Float32,
                    "train_dist_epoch": polars.Float32,
                    "val_loss_step": polars.Float32,
                    "val_loss_epoch": polars.Float32,
                    "val_dist_step": polars.Float32,
                    "val_dist_epoch": polars.Float32
                }),
            metric_files
        ), how="vertical"
    )
    
    df_train = df.select(
        polars.col("epoch"),
        polars.col("step"),
        polars.col("train_loss_step"),
        polars.col("train_dist_step")
    )
    df_train = df_train.filter(polars.col("epoch") <= last_epoch)
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
    
    df = df_train.join(df_validation, on="step", how="outer", validate="m:m")
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
    return df.collect()

def compute_agg_metrics(metric_files: List[str], last_epoch: int) -> polars.DataFrame:
    df = polars.concat(
        map(
            lambda metric_file: \
            polars.scan_csv(metric_file, has_header=True, separator=",")\
                .cast({
                    "epoch": polars.UInt32,
                    "step": polars.UInt64,
                    "train_loss_step": polars.Float32,
                    "train_loss_epoch": polars.Float32,
                    "train_dist_step": polars.Float32,
                    "train_dist_epoch": polars.Float32,
                    "val_loss_step": polars.Float32,
                    "val_loss_epoch": polars.Float32,
                    "val_dist_step": polars.Float32,
                    "val_dist_epoch": polars.Float32
                }),
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
    df_train = df_train.filter(polars.col("epoch") <= last_epoch)
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
    df_validation = df_validation.filter(polars.col("epoch") <= last_epoch)
    df_validation = df_validation.drop_nulls(subset="epoch")
    df_validation = df_validation.drop_nulls(subset="val_loss_epoch")
    df_validation = df_validation.drop_nulls(subset="val_dist_epoch")
    
    df = df_train.join(df_validation, on="epoch", how="inner", validate="1:1")
    return df.collect()

def load_metrics(metrics_file: str) -> polars.DataFrame:
    return polars.scan_parquet(metrics_file).collect()

def save_metrics(metrics_df: polars.DataFrame, metrics_file: str) -> None:
    metrics_df.write_parquet(metrics_file)

def metrics_for_plot(
        metrics_file: str,
        stage: str
    ) -> Tuple[polars.DataFrame, polars.DataFrame]:
    df = polars.scan_parquet(metrics_file)

    df_loss = df.select(
        polars.col(stage),
        polars.col(f"train_loss_{stage}"),
        polars.col(f"val_loss_{stage}")
    )
    df_dist = df.select(
        polars.col(stage),
        polars.col(f"train_dist_{stage}"),
        polars.col(f"val_dist_{stage}")
    )

    return df_loss.collect(), df_dist.collect()

def plot(
        metrics_df: polars.DataFrame,
        stage: str,
        name: str
    ) -> plotly.graph_objects.Figure:
    fig = plotly.express.line(
        data_frame=metrics_df.to_pandas(),
        title=f"{name} {stage}",
        x=stage,
        y=[f"train_{name}_{stage}", f"val_{name}_{stage}"],
        labels={stage: stage, "value": name}
    )
    new_names = {
        f"train_{name}_{stage}": f"train {name}",
        f"val_{name}_{stage}": f"validation {name}"
    }
    fig.for_each_trace(lambda trace: trace.update(name=new_names[trace.name]))
    fig_params = fig.to_dict()
    fig_params["layout"]["legend"]["title"] = None
    fig_params["layout"]["autosize"] = False
    fig.update_layout(**fig_params["layout"])

    return fig
