from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
import tifffile
import zarr
from scipy import sparse
from zarr.storage import ZipStore

from xenium_app.io import BundlePaths


CELL_SUMMARY_COLUMNS = [
    "cell_centroid_x",
    "cell_centroid_y",
    "cell_area",
    "nucleus_centroid_x",
    "nucleus_centroid_y",
    "nucleus_area",
    "z_level",
    "nucleus_count",
]

CELL_SUMMARY_COLUMNS_BY_COUNT = {
    7: CELL_SUMMARY_COLUMNS[:-1],
    8: CELL_SUMMARY_COLUMNS,
}


@dataclass
class ImageData:
    channel_name: str
    path: Path
    image: np.ndarray
    image_channel_index: int
    pixel_size_um_x: float
    pixel_size_um_y: float
    display_scale: float


@dataclass
class MorphologySource:
    path: Path
    channel_names: list[str]


@dataclass
class TranscriptPreview:
    frame: pd.DataFrame
    total_transcripts: int


@dataclass
class RegistrationResult:
    transform_um_to_px: np.ndarray
    transform_source: str


def _open_zip_group(path: Path) -> zarr.Group:
    store = ZipStore(str(path), mode="r")
    return zarr.open_group(store=store, mode="r")


def decode_cell_ids(cell_id_array: np.ndarray) -> list[str]:
    translation = str.maketrans("0123456789abcdef", "abcdefghijklmnop")
    decoded: list[str] = []
    for prefix, suffix in np.asarray(cell_id_array):
        hex_prefix = format(int(prefix), "08x").translate(translation)
        decoded.append(f"{hex_prefix}-{int(suffix)}")
    return decoded


def list_morphology_images(morphology_focus_dir: Path) -> list[Path]:
    images = sorted(morphology_focus_dir.glob("*.ome.tif"))
    if not images:
        raise FileNotFoundError(f"No OME-TIFF images found in {morphology_focus_dir}")
    return images


def _channel_index_from_filename(image_path: Path) -> int | None:
    match = re.match(r"ch0*([0-9]+)_", image_path.name.lower())
    if not match:
        return None
    return int(match.group(1))


def _channel_names_from_ome(ome_metadata: str) -> list[str]:
    channel_names: list[str] = []
    token = 'Name="'
    start = 0
    while token in ome_metadata[start:]:
        left = ome_metadata.find(token, start)
        right = ome_metadata.find('"', left + len(token))
        if left == -1 or right == -1:
            break
        value = ome_metadata[left + len(token) : right]
        if value.lower().endswith(".ome.tif"):
            break
        channel_names.append(value)
        start = right + 1
    return channel_names


def _extract_2d_plane(array: np.ndarray, axes: str, preferred_channel_index: int | None = None) -> tuple[np.ndarray, int]:
    if array.ndim == 2:
        return array, 0

    if "C" in axes:
        channel_axis = axes.index("C")
        channel_count = array.shape[channel_axis]
        channel_index = preferred_channel_index if preferred_channel_index is not None else 0
        channel_index = max(0, min(channel_index, channel_count - 1))
        plane = np.take(array, indices=channel_index, axis=channel_axis)
        squeezed = np.squeeze(plane)
        if squeezed.ndim == 2:
            return squeezed, channel_index

    squeezed = np.squeeze(array)
    if squeezed.ndim == 2:
        return squeezed, 0

    raise ValueError(f"Unsupported morphology image shape {array.shape} with axes {axes}. Expected a 2D plane or channel-first stack.")


def _pick_image_level(series: tifffile.TiffPageSeries, max_dim: int) -> tuple[np.ndarray, float, int]:
    levels = getattr(series, "levels", None) or [series]
    chosen = levels[-1]
    base_height, _ = levels[0].shape[-2:]
    chosen_height, _ = chosen.shape[-2:]
    scale = base_height / chosen_height
    for level in levels:
        height, width = level.shape[-2:]
        if max(height, width) <= max_dim:
            chosen = level
            scale = base_height / height
            break
    image = chosen.asarray()
    plane, channel_index = _extract_2d_plane(image, chosen.axes)
    return plane, float(scale), channel_index


def _extract_pixel_sizes(tif: tifffile.TiffFile) -> tuple[float, float]:
    ome = tif.ome_metadata or ""
    physical_size_x = 1.0
    physical_size_y = 1.0
    marker_x = 'PhysicalSizeX="'
    marker_y = 'PhysicalSizeY="'
    if marker_x in ome:
        physical_size_x = float(ome.split(marker_x, 1)[1].split('"', 1)[0])
    if marker_y in ome:
        physical_size_y = float(ome.split(marker_y, 1)[1].split('"', 1)[0])
    return physical_size_x, physical_size_y


def load_morphology_image(
    image_path: Path,
    max_dim: int = 2048,
    channel_index: int | None = None,
    channel_name: str | None = None,
) -> ImageData:
    with tifffile.TiffFile(image_path) as tif:
        series = tif.series[0]
        levels = getattr(series, "levels", None) or [series]
        chosen = levels[-1]
        base_height, _ = levels[0].shape[-2:]
        chosen_height, _ = chosen.shape[-2:]
        display_scale = base_height / chosen_height
        for level in levels:
            height, width = level.shape[-2:]
            if max(height, width) <= max_dim:
                chosen = level
                display_scale = base_height / height
                break
        preferred_channel_index = channel_index
        if preferred_channel_index is None:
            preferred_channel_index = _channel_index_from_filename(image_path)
        image_array = chosen.asarray()
        image, image_channel_index = _extract_2d_plane(
            image_array,
            chosen.axes,
            preferred_channel_index=preferred_channel_index,
        )
        pixel_size_um_x, pixel_size_um_y = _extract_pixel_sizes(tif)
        channel_names = _channel_names_from_ome(tif.ome_metadata or "")

    resolved_channel_name = channel_name
    if resolved_channel_name is None:
        if 0 <= image_channel_index < len(channel_names):
            resolved_channel_name = channel_names[image_channel_index]
        else:
            resolved_channel_name = image_path.stem.replace(".ome", "")
    image = image.astype(np.float32)
    # Percentile-based normalisation: clip to [p1, p99] then scale to [0, 1].
    # Using per-image /max would collapse every channel to the same [0, 1]
    # range regardless of true signal intensity, destroying between-channel
    # brightness differences that downstream models rely on.
    _p1 = float(np.percentile(image, 1))
    _p99 = float(np.percentile(image, 99))
    _span = _p99 - _p1 if _p99 > _p1 else 1.0
    image = np.clip((image - _p1) / _span, 0.0, 1.0)

    return ImageData(
        channel_name=resolved_channel_name,
        path=image_path,
        image=image,
        image_channel_index=image_channel_index,
        pixel_size_um_x=pixel_size_um_x,
        pixel_size_um_y=pixel_size_um_y,
        display_scale=display_scale,
    )


def load_morphology_source(morphology_focus_dir: Path) -> MorphologySource:
    images = list_morphology_images(morphology_focus_dir)
    source_path = images[0]
    with tifffile.TiffFile(source_path) as tif:
        channel_names = _channel_names_from_ome(tif.ome_metadata or "")
        if not channel_names:
            series = tif.series[0]
            channel_count = series.shape[series.axes.index("C")] if "C" in series.axes else 1
            channel_names = [f"Channel {idx}" for idx in range(channel_count)]
    return MorphologySource(path=source_path, channel_names=channel_names)


def load_all_channel_images(bundle: BundlePaths, max_dim: int = 2048) -> list[ImageData]:
    """Load every fluorescence channel available in the Xenium bundle.

    For morphology_focus/ directories each OME-TIFF file is one channel.
    For single-file morphologies every channel index is loaded separately.
    Returns images sorted by channel index with DAPI first when present.
    """
    if bundle.morphology_focus_dir is not None:
        image_files = list_morphology_images(bundle.morphology_focus_dir)

        # For the morphology_focus_NNNN.ome.tif naming convention the channel
        # name is NOT encoded in the individual filename.  The first file's OME
        # header lists ALL channel names in order (one per file), so we read
        # them once and map them by file-list position.
        all_channel_names: list[str] = []
        if image_files and _channel_index_from_filename(image_files[0]) is None:
            with tifffile.TiffFile(image_files[0]) as _tif:
                all_channel_names = _channel_names_from_ome(_tif.ome_metadata or "")

        images: list[ImageData] = []
        for file_pos, path in enumerate(image_files):
            channel_index = _channel_index_from_filename(path)
            # For positional naming (morphology_focus_NNNN), use the channel
            # name from the master list and channel_index=0 (each file is
            # single-channel — index 0 is the only plane).
            if channel_index is None:
                channel_name = (
                    all_channel_names[file_pos]
                    if file_pos < len(all_channel_names)
                    else f"Channel {file_pos}"
                )
                images.append(
                    load_morphology_image(path, max_dim=max_dim, channel_index=0, channel_name=channel_name)
                )
            else:
                images.append(load_morphology_image(path, max_dim=max_dim, channel_index=channel_index))
        return images

    if bundle.morphology_image_path is not None:
        path = bundle.morphology_image_path
        with tifffile.TiffFile(path) as tif:
            series = tif.series[0]
            channel_count = series.shape[series.axes.index("C")] if "C" in series.axes else 1
            channel_names = _channel_names_from_ome(tif.ome_metadata or "")
            if not channel_names:
                channel_names = [f"Channel {i}" for i in range(channel_count)]
        images = []
        for idx in range(channel_count):
            name = channel_names[idx] if idx < len(channel_names) else f"Channel {idx}"
            images.append(load_morphology_image(path, max_dim=max_dim, channel_index=idx, channel_name=name))
        return images

    raise FileNotFoundError("No morphology source found in Xenium bundle.")


def load_morphology_source_from_bundle(bundle: BundlePaths) -> MorphologySource:
    if bundle.morphology_image_path is not None:
        source_path = bundle.morphology_image_path
        with tifffile.TiffFile(source_path) as tif:
            channel_names = _channel_names_from_ome(tif.ome_metadata or "")
            if not channel_names:
                series = tif.series[0]
                channel_count = series.shape[series.axes.index("C")] if "C" in series.axes else 1
                channel_names = [f"Channel {idx}" for idx in range(channel_count)]
        return MorphologySource(path=source_path, channel_names=channel_names)

    if bundle.morphology_focus_dir is not None:
        return load_morphology_source(bundle.morphology_focus_dir)

    raise FileNotFoundError("No morphology source found in Xenium bundle.")


def load_cells_table(bundle: BundlePaths) -> tuple[pd.DataFrame, np.ndarray | None]:
    group = _open_zip_group(bundle.cells_zarr_path)
    cell_ids = np.asarray(group["cell_id"])
    decoded_cell_ids = decode_cell_ids(cell_ids)
    summary = np.asarray(group["cell_summary"])
    column_count = summary.shape[1]
    columns = CELL_SUMMARY_COLUMNS_BY_COUNT.get(column_count)
    if columns is None:
        columns = CELL_SUMMARY_COLUMNS[:column_count]
    frame = pd.DataFrame(summary[:, : len(columns)], columns=columns, index=decoded_cell_ids)
    if "nucleus_count" not in frame.columns:
        frame["nucleus_count"] = np.nan
    frame.index.name = "cell_id"
    transform = None
    masks = group.get("masks")
    if masks is not None and "homogeneous_transform" in masks:
        transform = np.asarray(masks["homogeneous_transform"])
    return frame, transform


def load_cell_feature_matrix(bundle: BundlePaths) -> tuple[sparse.csr_matrix, pd.DataFrame]:
    group = _open_zip_group(bundle.cell_feature_matrix_zarr_path)["cell_features"]
    attrs = dict(group.attrs)
    cell_ids = decode_cell_ids(np.asarray(group["cell_id"]))
    feature_names = list(attrs["feature_keys"])
    feature_ids = list(attrs.get("feature_ids", feature_names))
    feature_types = list(attrs.get("feature_types", ["unknown"] * len(feature_names)))

    if "csc" in group:
        data = np.asarray(group["csc"]["data"])
        indices = np.asarray(group["csc"]["indices"])
        indptr = np.asarray(group["csc"]["indptr"])
        features_by_cells = sparse.csc_matrix(
            (data, indices, indptr),
            shape=(len(feature_names), len(cell_ids)),
        )
    else:
        data = np.asarray(group["data"])
        indices = np.asarray(group["indices"])
        indptr = np.asarray(group["indptr"])
        features_by_cells = sparse.csr_matrix(
            (data, indices, indptr),
            shape=(len(feature_names), len(cell_ids)),
        )

    matrix = features_by_cells.transpose().tocsr()
    var = pd.DataFrame(
        {
            "feature_id": feature_ids,
            "feature_name": feature_names,
            "feature_type": feature_types,
        },
        index=feature_names,
    )
    var.index.name = "feature_name"

    return matrix, var


def _iter_grid_groups(group: zarr.Group) -> list[zarr.Group]:
    grids = group["grids"]["0"]
    grid_keys = sorted(grids.group_keys(), key=lambda item: tuple(int(part) for part in item.split(",")))
    return [grids[key] for key in grid_keys]


def _read_optional_grid_array(grid: zarr.Group, key: str) -> np.ndarray | None:
    try:
        return np.asarray(grid[key])
    except Exception:
        return None


def load_transcript_preview(bundle: BundlePaths, max_points: int = 20000) -> TranscriptPreview:
    group = _open_zip_group(bundle.transcripts_zarr_path)
    gene_names = list(group.attrs.get("gene_names", []))

    chunks: list[pd.DataFrame] = []
    total = 0
    for grid in _iter_grid_groups(group):
        locations = np.asarray(grid["location"])
        total += len(locations)
        if len(locations) == 0:
            continue
        genes = np.asarray(grid["gene_identity"])[:, 0]
        qv = np.asarray(grid["quality_score"]).reshape(-1)
        valid_values = _read_optional_grid_array(grid, "valid")
        if valid_values is None:
            valid = np.ones(len(locations), dtype=bool)
        else:
            valid = valid_values.reshape(-1).astype(bool)
        gene_labels = [
            gene_names[idx] if 0 <= idx < len(gene_names) else "unassigned"
            for idx in genes
        ]
        chunks.append(
            pd.DataFrame(
                {
                    "x_um": locations[:, 0],
                    "y_um": locations[:, 1],
                    "z_um": locations[:, 2],
                    "quality_score": qv,
                    "valid": valid,
                    "gene": gene_labels,
                }
            )
        )

    if not chunks:
        return TranscriptPreview(frame=pd.DataFrame(columns=["x_um", "y_um", "z_um", "quality_score", "valid", "gene"]), total_transcripts=0)

    frame = pd.concat(chunks, ignore_index=True)
    frame = frame.loc[frame["valid"]].reset_index(drop=True)
    if len(frame) > max_points:
        frame = frame.sample(max_points, random_state=0).sort_index().reset_index(drop=True)
    return TranscriptPreview(frame=frame, total_transcripts=total)


def _transform_points(transform: np.ndarray, x_um: np.ndarray, y_um: np.ndarray) -> np.ndarray:
    points = np.column_stack(
        [
            x_um,
            y_um,
            np.zeros_like(x_um, dtype=np.float32),
            np.ones_like(x_um, dtype=np.float32),
        ]
    )
    transformed = points @ transform.T
    return transformed[:, :2]


def build_registration(transform: np.ndarray | None, image: ImageData) -> RegistrationResult:
    if transform is not None and transform.shape == (4, 4):
        scale = np.eye(4, dtype=np.float32)
        scale[0, 0] = 1.0 / image.display_scale
        scale[1, 1] = 1.0 / image.display_scale
        return RegistrationResult(
            transform_um_to_px=scale @ transform,
            transform_source="cells.zarr.zip homogeneous_transform",
        )

    fallback = np.eye(4, dtype=np.float32)
    fallback[0, 0] = 1.0 / (image.pixel_size_um_x * image.display_scale)
    fallback[1, 1] = 1.0 / (image.pixel_size_um_y * image.display_scale)
    return RegistrationResult(
        transform_um_to_px=fallback,
        transform_source="OME physical pixel size fallback",
    )


def register_cells(cells: pd.DataFrame, registration: RegistrationResult) -> pd.DataFrame:
    xy = _transform_points(
        registration.transform_um_to_px,
        cells["cell_centroid_x"].to_numpy(dtype=np.float32),
        cells["cell_centroid_y"].to_numpy(dtype=np.float32),
    )
    registered = cells.copy()
    registered["x_px"] = xy[:, 0]
    registered["y_px"] = xy[:, 1]
    return registered


def register_transcripts(transcripts: pd.DataFrame, registration: RegistrationResult) -> pd.DataFrame:
    if transcripts.empty:
        return transcripts.assign(x_px=pd.Series(dtype=float), y_px=pd.Series(dtype=float))
    xy = _transform_points(
        registration.transform_um_to_px,
        transcripts["x_um"].to_numpy(dtype=np.float32),
        transcripts["y_um"].to_numpy(dtype=np.float32),
    )
    registered = transcripts.copy()
    registered["x_px"] = xy[:, 0]
    registered["y_px"] = xy[:, 1]
    return registered


def build_h5ad(
    bundle: BundlePaths,
    image: ImageData,
    registration: RegistrationResult,
    extra_obs: pd.DataFrame | None = None,
    obs_name_map: dict[str, str] | None = None,
    extra_images: list[ImageData] | None = None,
) -> ad.AnnData:
    cells, transform = load_cells_table(bundle)
    matrix, var = load_cell_feature_matrix(bundle)
    active_registration = registration if transform is not None else build_registration(None, image)
    registered_cells = register_cells(cells, active_registration)

    if matrix.shape[0] != len(registered_cells):
        raise ValueError("Cell-feature matrix cell count does not match cells.zarr.zip cell count.")

    obs = registered_cells.copy()
    obs["image_channel"] = image.channel_name
    if extra_obs is not None and not extra_obs.empty:
        obs = obs.join(extra_obs, how="left")
    if obs_name_map:
        obs = obs.rename(columns=obs_name_map)

    adata = ad.AnnData(X=matrix, obs=obs, var=var)
    x_px_col = obs_name_map.get("x_px", "x_px") if obs_name_map else "x_px"
    y_px_col = obs_name_map.get("y_px", "y_px") if obs_name_map else "y_px"
    cx_col = obs_name_map.get("cell_centroid_x", "cell_centroid_x") if obs_name_map else "cell_centroid_x"
    cy_col = obs_name_map.get("cell_centroid_y", "cell_centroid_y") if obs_name_map else "cell_centroid_y"
    adata.obsm["spatial"] = obs[[x_px_col, y_px_col]].to_numpy(dtype=np.float32)
    adata.obsm["spatial_microns"] = obs[[cx_col, cy_col]].to_numpy(dtype=np.float32)
    images_dict: dict[str, np.ndarray] = {"hires": image.image}
    channel_sources: dict[str, str] = {image.channel_name: str(image.path)}
    if extra_images:
        for extra in extra_images:
            key = re.sub(r"[^a-zA-Z0-9_]", "_", extra.channel_name).strip("_").lower()
            images_dict[key] = extra.image
            channel_sources[extra.channel_name] = str(extra.path)
    adata.uns["spatial"] = {
        bundle.manifest.get("run_name", "xenium_bundle"): {
            "images": images_dict,
            "scalefactors": {
                "tissue_hires_scalef": 1.0 / image.display_scale,
                "pixel_size_um_x": image.pixel_size_um_x,
                "pixel_size_um_y": image.pixel_size_um_y,
                "display_scale": image.display_scale,
            },
            "metadata": {
                "source_image": str(image.path),
                "channel_sources": channel_sources,
                "registration_source": active_registration.transform_source,
                "manifest_path": str(bundle.manifest_path),
            },
        }
    }
    return adata
