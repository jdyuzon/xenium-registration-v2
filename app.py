from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st
import re

from xenium_app.io import load_bundle_paths
from xenium_app.plotting import make_interactive_overlay_figure, plot_area_histogram, plot_tiff_only_zoom
from xenium_app.xenium import (
    build_h5ad,
    build_registration,
    load_all_channel_images,
    load_cells_table,
    load_morphology_image,
    load_morphology_source_from_bundle,
    load_transcript_preview,
    register_cells,
    register_transcripts,
)


st.set_page_config(page_title="Xenium Image Registration", layout="wide")
st.title("Xenium Image Registration")
st.caption("Load a Xenium Explorer subset bundle, preview the image-to-transcript registration, and export an h5ad.")

import sys as _sys
_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_PROJECT_ROOT))


@st.cache_data(show_spinner=False)
def cached_bundle(path: str):
    return load_bundle_paths(path)


@st.cache_data(show_spinner=False)
def cached_cells(path: str):
    bundle = load_bundle_paths(path)
    return load_cells_table(bundle)


@st.cache_data(show_spinner=False)
def cached_transcripts(path: str, max_points: int):
    bundle = load_bundle_paths(path)
    return load_transcript_preview(bundle, max_points=max_points)


@st.cache_data(show_spinner=False)
def cached_morphology_source(path: str):
    bundle = load_bundle_paths(path)
    return load_morphology_source_from_bundle(bundle)


@st.cache_data(show_spinner=False)
def cached_image(path: str, image_path: str, max_dim: int, channel_index: int | None = None, channel_name: str | None = None):
    _ = path
    return load_morphology_image(
        Path(image_path),
        max_dim=max_dim,
        channel_index=channel_index,
        channel_name=channel_name,
    )


def parse_cell_type_csv(uploaded_file) -> pd.DataFrame:
    table = pd.read_csv(uploaded_file)
    if table.shape[1] < 2:
        raise ValueError("CSV must have at least two columns, including cell_id and one label column.")

    first_col = table.columns[0]
    label_candidates = [
        "cell_type",
        "immune_focus",
        "five_class",
        "vision_cell_type",
        "harmonized_cell_type",
        "group",
    ]
    label_col = next((column for column in label_candidates if column in table.columns[1:]), table.columns[1])

    parsed = table[[first_col, label_col]].copy()
    parsed.columns = ["cell_id", "cell_type"]
    parsed["cell_id"] = parsed["cell_id"].astype(str)
    parsed["cell_type"] = parsed["cell_type"].astype(str)
    parsed = parsed.dropna(subset=["cell_id"]).drop_duplicates(subset=["cell_id"], keep="first")
    return parsed.set_index("cell_id")


def sanitize_metadata_name(name: str) -> str:
    cleaned = re.sub(r"[^0-9A-Za-z_]+", "_", name.strip())
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    if not cleaned:
        cleaned = "field"
    if cleaned[0].isdigit():
        cleaned = f"field_{cleaned}"
    return cleaned


def sanitize_filename_stem(name: str) -> str:
    return sanitize_metadata_name(name)


def build_obs_export_frame(registered_cells: pd.DataFrame, cell_type_annotations: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str]]:
    obs = registered_cells.join(cell_type_annotations, how="left")
    rename_map: dict[str, str] = {}
    used: set[str] = set()
    for column in obs.columns:
        base = sanitize_metadata_name(column)
        candidate = base
        suffix = 2
        while candidate in used:
            candidate = f"{base}_{suffix}"
            suffix += 1
        rename_map[column] = candidate
        used.add(candidate)
    return obs.rename(columns=rename_map), rename_map


def compute_zoom_ranges(center_x: float, center_y: float, half_width: float, image_width: int, image_height: int):
    x0 = max(0.0, center_x - half_width)
    x1 = min(float(image_width), center_x + half_width)
    y0 = max(0.0, center_y - half_width)
    y1 = min(float(image_height), center_y + half_width)
    return (x0, x1), (y1, y0)


def um_half_width_to_pixels(half_width_um: float, pixel_size_um_x: float, pixel_size_um_y: float) -> float:
    mean_pixel_size = (float(pixel_size_um_x) + float(pixel_size_um_y)) / 2.0
    return half_width_um / mean_pixel_size


def extract_selected_points(selection) -> list:
    if selection is None:
        return []
    if isinstance(selection, dict):
        if "selection" in selection and isinstance(selection["selection"], dict):
            return selection["selection"].get("points", []) or []
        return selection.get("points", []) or []
    selection_state = getattr(selection, "selection", None)
    if selection_state is not None:
        if isinstance(selection_state, dict):
            return selection_state.get("points", []) or []
        return getattr(selection_state, "points", []) or []
    return []


bundle_input = st.text_input(
    "Xenium bundle path",
    placeholder="/path/to/unzipped_xenium_bundle or /path/to/experiment.xenium",
)

with st.sidebar:
    st.header("Preview controls")
    max_image_dim = st.slider("Preview max image dimension", min_value=512, max_value=4096, value=2048, step=256)
    max_transcripts = st.slider("Max transcript points", min_value=1000, max_value=100000, value=20000, step=1000)
    cell_type_file = st.file_uploader("Cell type CSV", type=["csv"], help="First column: cell_id. Second column: cell_type.")
    zoom_half_width = st.slider("Zoom half-width (pixels)", min_value=50, max_value=4000, value=400, step=50)
    focus_cell_id = st.text_input("Focus cell_id", value="")

if bundle_input:
    try:
        bundle = cached_bundle(bundle_input)
        morphology_source = cached_morphology_source(bundle_input)
        channel_names = morphology_source.channel_names
        dapi_index = next((idx for idx, name in enumerate(channel_names) if "dapi" in name.lower()), 0)
        image = cached_image(
            bundle_input,
            str(morphology_source.path),
            max_image_dim,
            channel_index=dapi_index,
            channel_name=channel_names[dapi_index],
        )
        all_channel_images = load_all_channel_images(bundle, max_dim=max_image_dim)
        extra_images = [img for img in all_channel_images if img.channel_name.lower() != image.channel_name.lower()]
        cells, transform = cached_cells(bundle_input)
        transcript_preview = cached_transcripts(bundle_input, max_transcripts)

        registration = build_registration(transform, image)
        registered_cells = register_cells(cells, registration)
        registered_transcripts = register_transcripts(transcript_preview.frame, registration)
        cell_type_annotations = pd.DataFrame(index=registered_cells.index)
        matched_cell_types = 0
        if cell_type_file is not None:
            cell_type_annotations = parse_cell_type_csv(cell_type_file)
            matched_cell_types = int(cell_type_annotations.index.isin(registered_cells.index).sum())
            cell_type_annotations = cell_type_annotations.reindex(registered_cells.index)

        with st.sidebar:
            overlay_mode = st.selectbox(
                "Overlay layer",
                options=["Transcripts", "Cell centroids", "Cell types"],
            )

        export_obs_preview, obs_name_map = build_obs_export_frame(registered_cells, cell_type_annotations)
        overlay_cells = registered_cells.join(cell_type_annotations, how="left")
        overlay_cells["cell_area_px2"] = overlay_cells["cell_area"] / (image.pixel_size_um_x * image.pixel_size_um_y)

        if "focus_x_px" not in st.session_state:
            st.session_state["focus_x_px"] = image.image.shape[1] / 2.0
        if "focus_y_px" not in st.session_state:
            st.session_state["focus_y_px"] = image.image.shape[0] / 2.0
        effective_zoom_half_width = float(zoom_half_width)
        if focus_cell_id:
            match = overlay_cells.loc[overlay_cells.index.astype(str) == focus_cell_id]
            if not match.empty:
                st.session_state["focus_x_px"] = float(match.iloc[0]["x_px"])
                st.session_state["focus_y_px"] = float(match.iloc[0]["y_px"])
                effective_zoom_half_width = um_half_width_to_pixels(
                    half_width_um=250.0,
                    pixel_size_um_x=image.pixel_size_um_x * image.display_scale,
                    pixel_size_um_y=image.pixel_size_um_y * image.display_scale,
                )

        zoom_x_range, zoom_y_range = compute_zoom_ranges(
            center_x=float(st.session_state["focus_x_px"]),
            center_y=float(st.session_state["focus_y_px"]),
            half_width=effective_zoom_half_width,
            image_width=image.image.shape[1],
            image_height=image.image.shape[0],
        )

        metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(4)
        metrics_col1.metric("Cells", f"{len(cells):,}")
        metrics_col2.metric("Transcript preview", f"{len(registered_transcripts):,}")
        metrics_col3.metric("Total transcripts", f"{transcript_preview.total_transcripts:,}")
        metrics_col4.metric("Annotated cells", f"{matched_cell_types:,}" if cell_type_file is not None else "None")

        preview_col, summary_col = st.columns([3, 1])
        with preview_col:
            st.subheader(f"DAPI overlay: {overlay_mode}")
            interactive_overlay = make_interactive_overlay_figure(
                image=image,
                transcripts=registered_transcripts,
                cells=overlay_cells,
                overlay_mode=overlay_mode,
                x_range=zoom_x_range,
                y_range=zoom_y_range,
            )
            selection = st.plotly_chart(
                interactive_overlay,
                key="dapi_overlay_plot",
                on_select="rerun",
                selection_mode=("points",),
                use_container_width=False,
            )
            selected_points = extract_selected_points(selection)
            if selected_points:
                point = selected_points[0]
                st.session_state["focus_x_px"] = float(point["x"])
                st.session_state["focus_y_px"] = float(point["y"])
                if "customdata" in point and point["customdata"]:
                    candidate = point["customdata"][0]
                    if isinstance(candidate, str) and candidate in overlay_cells.index:
                        focus_cell_id = candidate

            st.subheader("Morphology channels")
            for channel_index, channel_name in enumerate(channel_names):
                morphology_image = cached_image(
                    bundle_input,
                    str(morphology_source.path),
                    max_image_dim,
                    channel_index=channel_index,
                    channel_name=channel_name,
                )
                st.caption(f"Channel {channel_index}: {channel_name}")
                st.pyplot(
                    plot_tiff_only_zoom(
                        image=morphology_image,
                        x_range=zoom_x_range,
                        y_range=zoom_y_range,
                    ),
                    use_container_width=False,
                )

        with summary_col:
            st.subheader("Bundle summary")
            st.write(f"Manifest: `{bundle.manifest_path}`")
            if bundle.morphology_focus_dir is not None:
                st.write(f"Morphology directory: `{bundle.morphology_focus_dir}`")
            if bundle.morphology_image_path is not None:
                st.write(f"Morphology image: `{bundle.morphology_image_path}`")
            st.write(f"Morphology source: `{morphology_source.path.name}`")
            st.write(f"DAPI channel: `{image.channel_name}`")
            st.write(f"Morphology channels: `{len(channel_names)}`")
            st.write(f"Display scale: `{image.display_scale:.3f}`")
            st.write(f"Pixel size (um): `{image.pixel_size_um_x:.4f}`, `{image.pixel_size_um_y:.4f}`")
            st.write("Cell size fields: `cell_area` and `nucleus_area` from Xenium `cells.zarr.zip`.")
            st.write("Cell size units: `cell_area` and `nucleus_area` are in `um^2`; centroids are in `um`.")
            st.write(f"Zoom center x_px: `{float(st.session_state['focus_x_px']):.1f}`")
            st.write(f"Zoom center y_px: `{float(st.session_state['focus_y_px']):.1f}`")
            if focus_cell_id:
                st.write("Focus window: `500 x 500 um` patch centered on the requested cell.")
            st.write(f"Analysis summary present: `{bundle.analysis_summary_path is not None}`")
            st.write(f"Analysis zarr present: `{bundle.analysis_zarr_path is not None}`")
            if cell_type_file is not None:
                st.write(f"Cell type matches: `{matched_cell_types}`")

            export_stem = sanitize_filename_stem(bundle.manifest.get("run_name", "xenium_registered"))
            export_name = f"{export_stem}.h5ad"
            output_path = st.text_input(
                "Export h5ad path",
                value=str(bundle.bundle_dir / export_name),
            )
            if st.button("Export h5ad", type="primary"):
                adata = build_h5ad(
                    bundle,
                    image,
                    registration,
                    extra_obs=cell_type_annotations,
                    obs_name_map=obs_name_map,
                    extra_images=extra_images,
                )
                adata.write_h5ad(output_path)
                st.success(f"Saved h5ad to {output_path}")

        st.subheader("h5ad metadata preview")
        st.caption("These are the `obs` columns that will be written to the h5ad file after sanitizing names.")
        st.dataframe(export_obs_preview.head(50), use_container_width=True)

        hist_col1, hist_col2 = st.columns(2)
        with hist_col1:
            st.pyplot(
                plot_area_histogram(
                    overlay_cells,
                    value_col="cell_area",
                    title="Cell area histogram (um^2) by cell_type",
                ),
                use_container_width=True,
            )
        with hist_col2:
            st.pyplot(
                plot_area_histogram(
                    overlay_cells,
                    value_col="cell_area_px2",
                    title="Cell area histogram (pixel^2) by cell_type",
                ),
                use_container_width=True,
            )

    except Exception as exc:
        st.error(str(exc))
else:
    st.info("Enter the path to an unzipped Xenium Explorer subset bundle to begin.")


# ---------------------------------------------------------------------------
# Cell type prediction tab
# ---------------------------------------------------------------------------

st.divider()
st.header("Cell Type Prediction")
st.caption(
    "Apply a trained five-class model to an existing h5ad to predict cell types. "
    "Features are extracted directly from the images stored in the h5ad — no original bundle needed."
)

pred_col1, pred_col2 = st.columns(2)
with pred_col1:
    pred_h5ad_path = st.text_input(
        "Input h5ad path",
        placeholder="/path/to/your_sample.h5ad",
        key="pred_h5ad",
    )
with pred_col2:
    pred_model_path = st.text_input(
        "Model path (.joblib)",
        placeholder="/path/to/xenium-ml-v2/focus_label_sets/five_class/model_outputs_dinov2_crop_combined/fast_five_class_model.joblib",
        key="pred_model",
    )

pred_output_dir = st.text_input(
    "Output directory",
    value=str(Path.home() / "xenium_predictions"),
    key="pred_output",
)

if st.button("Run cell type prediction", type="primary", key="pred_run"):
    if not pred_h5ad_path or not pred_model_path:
        st.warning("Please provide both an h5ad path and a model path.")
    else:
        try:
            with st.spinner("Loading model…"):
                from xenium_ml.fast_model import load_model_artifact, predict_h5ad as _predict_h5ad
                import anndata as _ad
                artifact = load_model_artifact(Path(pred_model_path).expanduser().resolve())

            with st.spinner("Extracting features and predicting…"):
                predictions = _predict_h5ad(pred_h5ad_path, artifact)

            out_dir = Path(pred_output_dir).expanduser().resolve()
            out_dir.mkdir(parents=True, exist_ok=True)
            stem = Path(pred_h5ad_path).stem
            csv_path = out_dir / "predictions.csv"
            h5ad_out = out_dir / f"{stem}_predicted.h5ad"

            predictions.to_csv(csv_path)
            adata_pred = _ad.read_h5ad(pred_h5ad_path)
            for col in predictions.columns:
                adata_pred.obs[col] = predictions.reindex(adata_pred.obs_names)[col]
            adata_pred.write_h5ad(h5ad_out)

            st.success(f"Predictions written to `{out_dir}`")

            # Quick summary
            vc = predictions["predicted_cell_type"].value_counts().reset_index()
            vc.columns = ["cell_type", "count"]
            vc["fraction"] = (vc["count"] / vc["count"].sum()).round(3)
            st.subheader("Predicted cell type distribution")
            st.dataframe(vc, use_container_width=True)

            # Show mean confidence per class
            prob_cols = [c for c in predictions.columns if c.startswith("prob_")]
            if prob_cols:
                mean_conf = predictions.groupby("predicted_cell_type")[prob_cols].mean().round(3)
                st.subheader("Mean class probabilities per predicted type")
                st.dataframe(mean_conf, use_container_width=True)

        except Exception as pred_exc:
            st.error(f"Prediction failed: {pred_exc}")
