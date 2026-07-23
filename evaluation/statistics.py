from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


# 标准的 8 列 metrics.csv schema（由 experiments.common.save_result 写出）。
# 其他列（topk/learners/strategy 等）来自 ablation 实验的 summary 表，混入
# 这里只会污染 dtype 并让 groupby mean 失败。
METRIC_COLUMNS = (
    "accuracy",
    "macro_f1",
    "auc",
    "normal_recall",
    "risk_recall",
    "threshold",
    "stage",
    "model",
)
_NUMERIC_COLUMNS = (
    "accuracy",
    "macro_f1",
    "auc",
    "normal_recall",
    "risk_recall",
    "threshold",
    "stage",
)


def _coerce_numeric(frame: pd.DataFrame) -> pd.DataFrame:
    """把数值列强制转成 float，无法解析的设为 NaN（防御半行残文件）。"""
    for column in _NUMERIC_COLUMNS:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def collect_metric_files(results_dir: str | Path) -> pd.DataFrame:
    frames = []
    for path in Path(results_dir).rglob("metrics.csv"):
        try:
            frame = pd.read_csv(path)
        except (pd.errors.EmptyDataError, UnicodeDecodeError):
            # 半行 / 截断文件：跳过，不污染下游
            continue
        if frame.empty:
            continue
        # 只保留标准 8 列；缺列的补 NaN（不是必需的，容错）
        frame = frame.reindex(columns=list(METRIC_COLUMNS))
        # 半行残文件会让某列 dtype 升级到 object，这里强制转回 float
        frame = _coerce_numeric(frame)
        frame["source"] = str(path)
        frames.append(frame)
    if not frames:
        return pd.DataFrame(columns=list(METRIC_COLUMNS) + ["source"])
    combined = pd.concat(frames, ignore_index=True)
    # dropna(subset=...) 会同时丢掉：stage 缺失的行（半行残文件）、全 NaN 的空行
    combined = combined.dropna(subset=["stage", "model"])
    return combined


def summarize_by_stage(metrics: pd.DataFrame) -> pd.DataFrame:
    if metrics.empty:
        return metrics
    return (
        metrics.groupby(["stage", "model"], as_index=False)
        .agg(
            accuracy=("accuracy", "mean"),
            macro_f1=("macro_f1", "mean"),
            auc=("auc", "mean"),
            normal_recall=("normal_recall", "mean"),
            risk_recall=("risk_recall", "mean"),
        )
        .sort_values(["stage", "macro_f1"], ascending=[True, False])
        .reset_index(drop=True)
    )
