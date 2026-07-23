from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def save_metric_heatmap(
    metrics: pd.DataFrame,
    value: str,
    output_path: str | Path,
    index: str = "model",
    columns: str = "stage",
) -> Path:
    table = metrics.pivot_table(index=index, columns=columns, values=value, aggfunc="mean")
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 6))
    sns.heatmap(table, annot=True, fmt=".4f", cmap="viridis")
    plt.tight_layout()
    plt.savefig(output, dpi=300)
    plt.close()
    return output


def save_stage_lineplot(
    metrics: pd.DataFrame,
    value: str,
    output_path: str | Path,
    hue: str = "model",
) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 5))
    sns.lineplot(data=metrics, x="stage", y=value, hue=hue, marker="o")
    plt.tight_layout()
    plt.savefig(output, dpi=300)
    plt.close()
    return output
