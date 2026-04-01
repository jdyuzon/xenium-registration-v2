from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from xenium_app.xenium import ImageData


def _resolve_ranges(image: ImageData, x_range: tuple[float, float] | None = None, y_range: tuple[float, float] | None = None):
    if x_range is None:
        x_range = (0, float(image.image.shape[1]))
    if y_range is None:
        y_range = (float(image.image.shape[0]), 0.0)
    return x_range, y_range


def _base_axes(image: ImageData, x_range: tuple[float, float] | None = None, y_range: tuple[float, float] | None = None):
    fig, ax = plt.subplots(figsize=(8, 8))
    x_range, y_range = _resolve_ranges(image, x_range=x_range, y_range=y_range)
    ax.set_xlabel("X (pixels)")
    ax.set_ylabel("Y (pixels)")
    ax.set_xlim(*x_range)
    ax.set_ylim(*y_range)
    ax.set_aspect("equal")
    return fig, ax


def _channel_color(image: ImageData) -> tuple[float, float, float]:
    name = image.channel_name.lower()
    if "dapi" in name:
        return (0.3, 0.55, 1.0)
    if "18s" in name:
        return (1.0, 0.75, 0.2)
    if "vimentin" in name or "alphasma" in name:
        return (0.95, 0.35, 0.35)
    if "cd45" in name or "e-cadherin" in name or "atp1a1" in name:
        return (0.2, 0.8, 0.45)
    palette = [
        (0.3, 0.55, 1.0),
        (0.2, 0.8, 0.45),
        (1.0, 0.75, 0.2),
        (0.95, 0.35, 0.35),
    ]
    return palette[image.image_channel_index % len(palette)]


def _colorize_image(image: ImageData) -> np.ndarray:
    data = image.image.astype(np.float32)
    low = float(np.quantile(data, 0.02))
    high = float(np.quantile(data, 0.995))
    if high <= low:
        scaled = np.clip(data, 0.0, 1.0)
    else:
        scaled = np.clip((data - low) / (high - low), 0.0, 1.0)

    rgb = np.zeros((*scaled.shape, 3), dtype=np.float32)
    color = np.asarray(_channel_color(image), dtype=np.float32)
    rgb[:] = scaled[..., None] * color
    return rgb


def plot_overlay(
    image: ImageData,
    transcripts: pd.DataFrame,
    cells: pd.DataFrame,
    show_cells: bool = True,
    show_transcripts: bool = True,
    x_range: tuple[float, float] | None = None,
    y_range: tuple[float, float] | None = None,
):
    fig, ax = _base_axes(image, x_range=x_range, y_range=y_range)
    ax.imshow(image.image, cmap="gray", origin="upper")

    if show_transcripts and not transcripts.empty:
        ax.scatter(
            transcripts["x_px"],
            transcripts["y_px"],
            s=2,
            c="#ff6b35",
            alpha=0.35,
            linewidths=0,
            label="Transcripts",
        )

    if show_cells and not cells.empty:
        ax.scatter(
            cells["x_px"],
            cells["y_px"],
            s=10,
            c="#00c2a8",
            alpha=0.8,
            linewidths=0,
            label="Cell centroids",
        )

    ax.set_title(f"Registered overlay: {image.channel_name}")
    ax.legend(loc="upper right")
    fig.tight_layout()
    return fig


def plot_metadata_overlay(
    image: ImageData,
    cells: pd.DataFrame,
    field_name: str,
    field_label: str | None = None,
    x_range: tuple[float, float] | None = None,
    y_range: tuple[float, float] | None = None,
):
    fig, ax = _base_axes(image, x_range=x_range, y_range=y_range)
    ax.imshow(image.image, cmap="gray", origin="upper")
    points = ax.scatter(
        cells["x_px"],
        cells["y_px"],
        s=10,
        c=cells[field_name],
        cmap="viridis",
        alpha=0.85,
        linewidths=0,
    )
    label = field_label or field_name
    ax.set_title(f"Registered overlay: {label}")
    fig.colorbar(points, ax=ax, fraction=0.046, pad=0.04, label=label)
    fig.tight_layout()
    return fig


def plot_categorical_overlay(
    image: ImageData,
    cells: pd.DataFrame,
    field_name: str,
    field_label: str | None = None,
    x_range: tuple[float, float] | None = None,
    y_range: tuple[float, float] | None = None,
):
    fig, ax = _base_axes(image, x_range=x_range, y_range=y_range)
    ax.imshow(image.image, cmap="gray", origin="upper")
    valid = cells[field_name].notna()
    subset = cells.loc[valid].copy()
    if subset.empty:
        ax.set_title(f"Registered overlay: {field_label or field_name}")
        fig.tight_layout()
        return fig

    categories = pd.Categorical(subset[field_name].astype(str))
    points = ax.scatter(
        subset["x_px"],
        subset["y_px"],
        s=10,
        c=categories.codes,
        cmap="tab20",
        alpha=0.85,
        linewidths=0,
    )
    label = field_label or field_name
    ax.set_title(f"Registered overlay: {label}")
    handles, _ = points.legend_elements(num=len(categories.categories))
    ax.legend(handles, list(categories.categories), title=label, loc="upper right", fontsize=8)
    fig.tight_layout()
    return fig


def plot_tiff_only(image: ImageData):
    fig, ax = _base_axes(image)
    ax.imshow(_colorize_image(image), origin="upper")
    ax.set_title(f"TIFF only: {image.channel_name}")
    fig.tight_layout()
    return fig


def plot_tiff_only_zoom(
    image: ImageData,
    x_range: tuple[float, float] | None = None,
    y_range: tuple[float, float] | None = None,
):
    fig, ax = _base_axes(image, x_range=x_range, y_range=y_range)
    ax.imshow(_colorize_image(image), origin="upper")
    ax.set_title(f"TIFF only: {image.channel_name}")
    fig.tight_layout()
    return fig


def plot_transcripts_only(
    image: ImageData,
    transcripts: pd.DataFrame,
    cells: pd.DataFrame,
    show_cells: bool = True,
    show_transcripts: bool = True,
):
    fig, ax = _base_axes(image)

    if show_transcripts and not transcripts.empty:
        ax.scatter(
            transcripts["x_px"],
            transcripts["y_px"],
            s=2,
            c="#ff6b35",
            alpha=0.5,
            linewidths=0,
            label="Transcripts",
        )

    if show_cells and not cells.empty:
        ax.scatter(
            cells["x_px"],
            cells["y_px"],
            s=10,
            c="#00c2a8",
            alpha=0.85,
            linewidths=0,
            label="Cell centroids",
        )

    ax.set_title("Transcript layer")
    if show_cells or show_transcripts:
        ax.legend(loc="upper right")
    fig.tight_layout()
    return fig


def make_interactive_overlay_figure(
    image: ImageData,
    transcripts: pd.DataFrame,
    cells: pd.DataFrame,
    overlay_mode: str,
    x_range: tuple[float, float] | None = None,
    y_range: tuple[float, float] | None = None,
) -> go.Figure:
    x_range, y_range = _resolve_ranges(image, x_range=x_range, y_range=y_range)
    base_gray = (np.clip(image.image, 0.0, 1.0) * 255).astype(np.uint8)
    base = np.stack([base_gray, base_gray, base_gray], axis=-1)

    fig = go.Figure()
    fig.add_trace(go.Image(z=base, colormodel="rgb"))

    if overlay_mode == "Transcripts" and not transcripts.empty:
        fig.add_trace(
            go.Scattergl(
                x=transcripts["x_px"],
                y=transcripts["y_px"],
                mode="markers",
                marker={"size": 4, "color": "#ff6b35", "opacity": 0.35},
                customdata=np.stack(
                    [
                        transcripts["gene"].astype(str).to_numpy(),
                        transcripts["x_um"].to_numpy(),
                        transcripts["y_um"].to_numpy(),
                    ],
                    axis=1,
                ),
                hovertemplate="gene=%{customdata[0]}<br>x_um=%{customdata[1]:.2f}<br>y_um=%{customdata[2]:.2f}<extra></extra>",
                name="Transcripts",
            )
        )
    elif overlay_mode in {"Cell centroids", "Cell types"} and not cells.empty:
        if overlay_mode == "Cell types" and "cell_type" in cells.columns:
            categories = pd.Categorical(cells["cell_type"].fillna("unlabeled").astype(str))
            marker = {
                "size": 6,
                "color": np.asarray(categories.codes),
                "colorscale": "Turbo",
                "opacity": 0.8,
                "showscale": False,
            }
            customdata = np.stack(
                [
                    np.asarray(cells.index.astype(str)),
                    np.asarray(categories.astype(str)),
                ],
                axis=1,
            )
            hovertemplate = "cell_id=%{customdata[0]}<br>cell_type=%{customdata[1]}<extra></extra>"
            name = "Cell types"
        else:
            marker = {"size": 6, "color": "#00c2a8", "opacity": 0.8}
            customdata = np.stack([cells.index.astype(str).to_numpy()], axis=1)
            hovertemplate = "cell_id=%{customdata[0]}<extra></extra>"
            name = "Cell centroids"

        fig.add_trace(
            go.Scattergl(
                x=cells["x_px"],
                y=cells["y_px"],
                mode="markers",
                marker=marker,
                customdata=customdata,
                hovertemplate=hovertemplate,
                name=name,
            )
        )

    fig.update_layout(
        title=f"DAPI overlay: {overlay_mode}",
        width=800,
        height=800,
        dragmode="select",
        margin={"l": 40, "r": 20, "t": 50, "b": 40},
        xaxis={"range": list(x_range), "title": "X (pixels)"},
        yaxis={"range": list(y_range), "title": "Y (pixels)", "autorange": False},
        showlegend=False,
    )
    return fig


def plot_area_histogram(
    cells: pd.DataFrame,
    value_col: str,
    label_col: str = "cell_type",
    title: str = "",
):
    fig, ax = plt.subplots(figsize=(9, 4.5))
    if label_col in cells.columns and cells[label_col].notna().any():
        top_labels = cells[label_col].fillna("unlabeled").astype(str).value_counts().head(8).index.tolist()
        for label in top_labels:
            subset = cells.loc[cells[label_col].fillna("unlabeled").astype(str) == label, value_col].dropna()
            if len(subset) == 0:
                continue
            ax.hist(subset, bins=40, alpha=0.45, label=label)
        ax.legend(loc="upper right", fontsize=8)
    else:
        ax.hist(cells[value_col].dropna(), bins=40, color="#4c78a8", alpha=0.8)

    ax.set_title(title or value_col)
    ax.set_xlabel(value_col)
    ax.set_ylabel("Cell count")
    fig.tight_layout()
    return fig
