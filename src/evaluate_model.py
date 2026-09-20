"""
Model evaluation: metrics, comparison tables and report charts.

This module is deliberately separate from ``train_model.py`` so that the same
metric and plotting code can be reused to re-evaluate a saved model later
without retraining it.

Every number produced here comes from an actual model prediction on the held-out
test split.  Nothing is hard-coded.

Charts written to ``reports/``:
    confusion_matrix.png    - errors of the selected model, per class
    model_comparison.png    - the four algorithms side by side
    performance_metrics.png - per-class precision / recall / F1
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import matplotlib

# "Agg" is a non-interactive backend: it writes PNG files without needing a
# display server, which is required when running on a headless machine.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

# ---------------------------------------------------------------------------
# Allow this file to be run directly as `python src/evaluate_model.py` as well as via
# `python -m src.evaluate_model`. When run directly, Python puts src/ on the path
# instead of the project root, so `from src import ...` would fail. Adding the
# project root here makes both invocations work identically.
# ---------------------------------------------------------------------------
if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path

    _sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

from src import config

logger = logging.getLogger(__name__)

# ==========================================================================
# Chart palette
# ==========================================================================
# These hex values come from a validated categorical palette: every adjacent
# pair clears the colour-vision-deficiency separation threshold, so the charts
# stay readable for colourblind viewers.  Because two of the light-mode hues
# sit below 3:1 contrast against the surface, every bar also carries a visible
# numeric label - colour is never the only channel.
SERIES_COLORS: tuple[str, ...] = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")
STATUS_GOOD = "#0ca30c"    # REAL
STATUS_CRITICAL = "#d03b3b"  # FAKE

SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"

# Single-hue sequential ramp (light -> dark) used for the confusion matrix.
SEQUENTIAL_BLUE = LinearSegmentedColormap.from_list(
    "sequential_blue",
    ["#f4f8fe", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#104281"],
)

# The metrics reported in the comparison table, in display order.
METRIC_KEYS: tuple[str, ...] = ("accuracy", "precision", "recall", "f1")
METRIC_LABELS: tuple[str, ...] = ("Accuracy", "Precision", "Recall", "F1-score")

# FAKE is treated as the positive class: the purpose of the system is to catch
# fake news, so "recall" should mean "of all the fake articles, how many did we
# catch?".  This choice is stated in the README and should be stated in a viva.
POSITIVE_LABEL = config.LABEL_FAKE


# ==========================================================================
# Metrics
# ==========================================================================
def compute_metrics(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    positive_label: str = POSITIVE_LABEL,
) -> Dict[str, float]:
    """
    Compute the four headline metrics for a set of predictions.

    Precision, recall and F1 are reported for ``positive_label`` (FAKE by
    default).  ``zero_division=0`` stops scikit-learn from raising when a
    degenerate model predicts only one class.
    """
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(
            precision_score(y_true, y_pred, pos_label=positive_label, zero_division=0)
        ),
        "recall": float(
            recall_score(y_true, y_pred, pos_label=positive_label, zero_division=0)
        ),
        "f1": float(
            f1_score(y_true, y_pred, pos_label=positive_label, zero_division=0)
        ),
    }


def per_class_metrics(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    labels: Sequence[str] = config.CLASS_NAMES,
) -> Dict[str, Dict[str, float]]:
    """Return precision / recall / F1 for each class separately."""
    report = classification_report(
        y_true, y_pred, labels=list(labels), output_dict=True, zero_division=0
    )
    return {
        label: {
            "precision": float(report[label]["precision"]),
            "recall": float(report[label]["recall"]),
            "f1": float(report[label]["f1-score"]),
            "support": int(report[label]["support"]),
        }
        for label in labels
        if label in report
    }


def format_comparison_table(results: List[dict]) -> str:
    """
    Render the model comparison as a fixed-width text table.

    ``results`` is a list of dicts with a ``name`` key plus the metric keys.
    The table is printed at the end of training and pasted into the README.
    """
    header = (
        f"{'Model':<22}{'Accuracy':>10}{'Precision':>11}"
        f"{'Recall':>9}{'F1':>9}{'Train (s)':>11}"
    )
    lines = [header, "-" * len(header)]
    for row in sorted(results, key=lambda r: r["f1"], reverse=True):
        lines.append(
            f"{row['name']:<22}"
            f"{row['accuracy']:>10.4f}"
            f"{row['precision']:>11.4f}"
            f"{row['recall']:>9.4f}"
            f"{row['f1']:>9.4f}"
            f"{row.get('train_seconds', float('nan')):>11.2f}"
        )
    return "\n".join(lines)


# ==========================================================================
# Chart helpers
# ==========================================================================
def _style_axes(ax: plt.Axes, *, ygrid: bool = True) -> None:
    """Apply the shared chart styling: recessive grid, no heavy frame."""
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#c3c2b7")
        ax.spines[side].set_linewidth(1)
    if ygrid:
        ax.set_axisbelow(True)
        ax.yaxis.grid(True, color=GRIDLINE, linewidth=1)
        ax.xaxis.grid(False)
    ax.tick_params(colors=INK_MUTED, labelsize=9, length=0)


def _new_figure(width: float, height: float):
    """Create a figure/axes pair with the project's surface colour."""
    fig, ax = plt.subplots(figsize=(width, height), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    return fig, ax


def plot_confusion_matrix(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    model_name: str,
    output_path: Path | None = None,
    labels: Sequence[str] = config.CLASS_NAMES,
) -> Path:
    """
    Draw the confusion matrix as an annotated heatmap.

    Each cell shows the raw count and the row percentage, so the chart is
    readable without relying on the colour scale alone.
    """
    output_path = output_path or (config.REPORTS_DIR / "confusion_matrix.png")
    matrix = confusion_matrix(y_true, y_pred, labels=list(labels))
    row_totals = matrix.sum(axis=1, keepdims=True)
    # Guard against a divide-by-zero if a class is absent from the test split.
    percentages = np.divide(
        matrix, np.where(row_totals == 0, 1, row_totals), dtype=float
    ) * 100

    fig, ax = _new_figure(5.6, 4.8)
    image = ax.imshow(percentages, cmap=SEQUENTIAL_BLUE, vmin=0, vmax=100)

    ax.set_xticks(range(len(labels)), [f"Predicted\n{lab}" for lab in labels])
    ax.set_yticks(range(len(labels)), [f"Actual\n{lab}" for lab in labels])

    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            # Flip the label to white once the cell is dark enough for black
            # text to lose contrast.
            text_color = "#ffffff" if percentages[row, col] > 55 else INK_PRIMARY
            ax.text(
                col, row,
                f"{matrix[row, col]:,}\n{percentages[row, col]:.1f}%",
                ha="center", va="center",
                color=text_color, fontsize=13, fontweight="600", linespacing=1.4,
            )

    # A 2px surface-coloured gap between cells keeps adjacent fills separate.
    ax.set_xticks(np.arange(-0.5, len(labels), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(labels), 1), minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=2)
    ax.tick_params(which="minor", length=0)
    ax.tick_params(colors=INK_SECONDARY, labelsize=10, length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.set_title(
        f"Confusion Matrix - {model_name}",
        color=INK_PRIMARY, fontsize=13, fontweight="600", pad=14,
    )
    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label("% of actual class", color=INK_SECONDARY, fontsize=9)
    colorbar.ax.tick_params(colors=INK_MUTED, labelsize=8, length=0)
    colorbar.outline.set_visible(False)

    fig.tight_layout()
    fig.savefig(output_path, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", output_path)
    return output_path


def plot_model_comparison(
    results: List[dict], output_path: Path | None = None
) -> Path:
    """
    Grouped bar chart comparing every trained algorithm on all four metrics.

    Each bar is directly labelled with its value, which is what makes the chart
    readable for viewers who cannot distinguish the hues.
    """
    output_path = output_path or (config.REPORTS_DIR / "model_comparison.png")
    ordered = sorted(results, key=lambda r: r["f1"], reverse=True)
    names = [row["name"] for row in ordered]

    x_positions = np.arange(len(names))
    bar_width = 0.19

    fig, ax = _new_figure(10.5, 5.4)
    _style_axes(ax)

    for index, (key, label) in enumerate(zip(METRIC_KEYS, METRIC_LABELS)):
        values = [row[key] for row in ordered]
        offset = (index - 1.5) * bar_width
        bars = ax.bar(
            x_positions + offset, values,
            width=bar_width - 0.015,  # the gap keeps adjacent fills separated
            label=label, color=SERIES_COLORS[index],
            edgecolor=SURFACE, linewidth=1.5, zorder=3,
        )
        ax.bar_label(
            bars, fmt="%.3f", padding=3,
            color=INK_SECONDARY, fontsize=7.5, rotation=90,
        )

    ax.set_xticks(x_positions, names, color=INK_SECONDARY, fontsize=10)
    ax.set_ylim(0, 1.16)
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.set_ylabel("Score", color=INK_SECONDARY, fontsize=10)
    ax.set_title(
        "Model Comparison on the Held-out Test Set",
        color=INK_PRIMARY, fontsize=14, fontweight="600", pad=16, loc="left",
    )
    legend = ax.legend(
        ncol=4, frameon=False, loc="upper center",
        bbox_to_anchor=(0.5, -0.09), fontsize=9,
    )
    for text in legend.get_texts():
        text.set_color(INK_SECONDARY)

    fig.tight_layout()
    fig.savefig(output_path, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", output_path)
    return output_path


def plot_performance_metrics(
    class_metrics: Dict[str, Dict[str, float]],
    model_name: str,
    output_path: Path | None = None,
) -> Path:
    """
    Horizontal bars showing precision / recall / F1 for each class.

    This answers the question a confusion matrix only implies: is the model
    weaker at catching FAKE articles or at clearing REAL ones?
    """
    output_path = output_path or (config.REPORTS_DIR / "performance_metrics.png")
    classes = list(class_metrics.keys())
    metric_names = ("precision", "recall", "f1")
    metric_labels = ("Precision", "Recall", "F1-score")

    y_positions = np.arange(len(metric_names))
    bar_height = 0.34

    fig, ax = _new_figure(9.0, 4.2)
    _style_axes(ax, ygrid=False)
    ax.set_axisbelow(True)
    ax.xaxis.grid(True, color=GRIDLINE, linewidth=1)

    # FAKE and REAL are states, not arbitrary series, so they use the reserved
    # status colours - always alongside a text label, never colour alone.
    class_colors = {
        config.LABEL_FAKE: STATUS_CRITICAL,
        config.LABEL_REAL: STATUS_GOOD,
    }

    for index, class_name in enumerate(classes):
        values = [class_metrics[class_name][metric] for metric in metric_names]
        offset = (index - (len(classes) - 1) / 2) * bar_height
        bars = ax.barh(
            y_positions + offset, values,
            height=bar_height - 0.03,
            label=f"{class_name} (n={class_metrics[class_name]['support']:,})",
            color=class_colors.get(class_name, SERIES_COLORS[index]),
            edgecolor=SURFACE, linewidth=1.5, zorder=3,
        )
        ax.bar_label(
            bars, fmt="  %.3f", padding=2, color=INK_SECONDARY, fontsize=9
        )

    ax.set_yticks(y_positions, metric_labels, color=INK_SECONDARY, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.12)
    ax.set_xticks(np.arange(0, 1.01, 0.2))
    ax.set_xlabel("Score", color=INK_SECONDARY, fontsize=10)
    ax.set_title(
        f"Per-class Performance - {model_name}",
        color=INK_PRIMARY, fontsize=14, fontweight="600", pad=16, loc="left",
    )
    legend = ax.legend(frameon=False, loc="lower right", fontsize=9)
    for text in legend.get_texts():
        text.set_color(INK_SECONDARY)

    fig.tight_layout()
    fig.savefig(output_path, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", output_path)
    return output_path


def generate_all_reports(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    model_name: str,
    results: List[dict],
) -> List[Path]:
    """Generate every chart in ``reports/`` and return the written paths."""
    config.ensure_directories()
    return [
        plot_confusion_matrix(y_true, y_pred, model_name),
        plot_model_comparison(results),
        plot_performance_metrics(per_class_metrics(y_true, y_pred), model_name),
    ]


def load_saved_metrics() -> dict | None:
    """
    Read ``models/metrics.json``, or return None if training has not been run.

    The Flask dashboard uses this so it can display real evaluation numbers
    instead of placeholders.
    """
    try:
        with open(config.METRICS_PATH, encoding="utf-8") as handle:
            return json.load(handle)
    except FileNotFoundError:
        logger.warning("No metrics file at %s - run training first.",
                       config.METRICS_PATH)
        return None
    except json.JSONDecodeError as exc:
        logger.error("Metrics file is corrupt: %s", exc)
        return None


def _main() -> None:
    """Re-evaluate the saved model against the reproducible test split."""
    import joblib
    from sklearn.model_selection import train_test_split

    from src.dataset import build_processed_dataset
    from src.exceptions import ModelNotFoundError

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if not config.MODEL_PATH.is_file() or not config.VECTORIZER_PATH.is_file():
        raise ModelNotFoundError(
            "model.pkl / vectorizer.pkl not found. Run: python src/train_model.py"
        )

    model = joblib.load(config.MODEL_PATH)
    vectorizer = joblib.load(config.VECTORIZER_PATH)
    frame = build_processed_dataset()

    # Identical split parameters as training -> identical test set.
    _, x_test, _, y_test = train_test_split(
        frame["clean_text"], frame["label"],
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=frame["label"],
    )
    y_pred = model.predict(vectorizer.transform(x_test))

    print("\nClassification report:\n")
    print(classification_report(y_test, y_pred, zero_division=0))
    print("Headline metrics (positive class = FAKE):")
    for key, value in compute_metrics(y_test, y_pred).items():
        print(f"  {key:<10}: {value:.4f}")

    plot_confusion_matrix(y_test, y_pred, type(model).__name__)


if __name__ == "__main__":  # pragma: no cover
    _main()
