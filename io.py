from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class BundlePaths:
    bundle_dir: Path
    manifest_path: Path
    gene_panel_path: Path | None
    morphology_focus_dir: Path | None
    morphology_image_path: Path | None
    cells_zarr_path: Path
    cell_feature_matrix_zarr_path: Path
    transcripts_zarr_path: Path
    analysis_zarr_path: Path | None
    analysis_summary_path: Path | None
    manifest: dict[str, Any]


def _resolve_path(base_dir: Path, value: str | None, default: str | None = None) -> Path | None:
    candidate = value or default
    if not candidate:
        return None
    path = Path(candidate)
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()


def _resolve_user_input_path(bundle_input: str | Path) -> Path:
    raw = Path(bundle_input).expanduser()
    candidates: list[Path] = []

    if raw.is_absolute():
        candidates.append(raw)
    else:
        candidates.append((Path.cwd() / raw).resolve())
        candidates.append((Path.home() / raw).resolve())

    for candidate in candidates:
        if candidate.exists():
            return candidate

    return candidates[0]


def _pick_morphology_location(manifest: dict[str, Any], base_dir: Path) -> tuple[Path | None, Path | None]:
    images = manifest.get("images", {})
    dir_candidate_keys = (
        "morphology_focus",
        "morphology_focus_dir",
        "morphology_focus_images",
    )
    for key in dir_candidate_keys:
        path = _resolve_path(base_dir, images.get(key))
        if path and path.exists():
            if path.is_dir():
                return path, None
            if path.is_file():
                return None, path

    default_path = (base_dir / "morphology_focus").resolve()
    if default_path.exists():
        return default_path, None

    file_candidate_keys = (
        "morphology_focus_filepath",
        "morphology_mip_filepath",
        "morphology_filepath",
    )
    for key in file_candidate_keys:
        path = _resolve_path(base_dir, images.get(key))
        if path and path.exists() and path.is_file():
            return None, path

    for filename in ("morphology_focus.ome.tif", "morphology_mip.ome.tif", "morphology.ome.tif"):
        path = (base_dir / filename).resolve()
        if path.exists():
            return None, path
    return None, None


def load_bundle_paths(bundle_input: str | Path) -> BundlePaths:
    input_path = _resolve_user_input_path(bundle_input)
    manifest_path = input_path if input_path.name == "experiment.xenium" else input_path / "experiment.xenium"
    if not manifest_path.exists():
        home_candidate = (Path.home() / Path(bundle_input).expanduser()).resolve()
        raise FileNotFoundError(
            f"Could not find experiment.xenium at {manifest_path}. "
            f"If you meant your home Downloads folder, try {home_candidate}. "
            "Provide either the bundle directory or the manifest path."
        )

    bundle_dir = manifest_path.parent
    manifest = json.loads(manifest_path.read_text())
    xe_files = manifest.get("xenium_explorer_files", {})

    morphology_focus_dir, morphology_image_path = _pick_morphology_location(manifest, bundle_dir)
    if morphology_focus_dir is None and morphology_image_path is None:
        raise FileNotFoundError(
            "Could not locate morphology images from manifest or bundle directory. "
            "Expected either morphology_focus/ or a top-level morphology*.ome.tif file."
        )

    cells_zarr_path = _resolve_path(bundle_dir, xe_files.get("cells_zarr") or xe_files.get("cells_zarr_filepath"), "cells.zarr.zip")
    cell_feature_matrix_zarr_path = _resolve_path(
        bundle_dir,
        xe_files.get("cell_feature_matrix_zarr") or xe_files.get("cell_features_zarr_filepath"),
        "cell_feature_matrix.zarr.zip",
    )
    transcripts_zarr_path = _resolve_path(bundle_dir, xe_files.get("transcripts_zarr") or xe_files.get("transcripts_zarr_filepath"), "transcripts.zarr.zip")
    analysis_zarr_path = _resolve_path(bundle_dir, xe_files.get("analysis_zarr") or xe_files.get("analysis_zarr_filepath"), "analysis.zarr.zip")
    analysis_summary_path = _resolve_path(
        bundle_dir,
        xe_files.get("analysis_summary") or xe_files.get("analysis_summary_filepath"),
        "analysis_summary.html",
    )
    gene_panel_path = _resolve_path(bundle_dir, manifest.get("gene_panel"), "gene_panel.json")

    required_paths = {
        "cells.zarr.zip": cells_zarr_path,
        "cell_feature_matrix.zarr.zip": cell_feature_matrix_zarr_path,
        "transcripts.zarr.zip": transcripts_zarr_path,
    }
    for label, path in required_paths.items():
        if path is None or not path.exists():
            raise FileNotFoundError(f"Missing required Xenium bundle file: {label}")

    return BundlePaths(
        bundle_dir=bundle_dir,
        manifest_path=manifest_path,
        gene_panel_path=gene_panel_path if gene_panel_path and gene_panel_path.exists() else None,
        morphology_focus_dir=morphology_focus_dir,
        morphology_image_path=morphology_image_path,
        cells_zarr_path=cells_zarr_path,
        cell_feature_matrix_zarr_path=cell_feature_matrix_zarr_path,
        transcripts_zarr_path=transcripts_zarr_path,
        analysis_zarr_path=analysis_zarr_path if analysis_zarr_path and analysis_zarr_path.exists() else None,
        analysis_summary_path=analysis_summary_path if analysis_summary_path and analysis_summary_path.exists() else None,
        manifest=manifest,
    )
