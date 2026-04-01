from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import tifffile
import zarr
from zarr.storage import ZipStore

from xenium_app.io import load_bundle_paths
from xenium_app.xenium import (
    build_h5ad,
    build_registration,
    load_cell_feature_matrix,
    load_cells_table,
    load_morphology_image,
    load_transcript_preview,
    register_cells,
)


def _write_zip_group(path: Path) -> zarr.Group:
    store = ZipStore(str(path), mode="w")
    return zarr.open_group(store=store, mode="w")


def _build_fixture_bundle(bundle_dir: Path) -> Path:
    bundle_dir.mkdir(parents=True, exist_ok=True)
    morphology_dir = bundle_dir / "morphology_focus"
    morphology_dir.mkdir()

    image = np.arange(64, dtype=np.uint16).reshape(8, 8)
    tifffile.imwrite(
        morphology_dir / "ch0000_dapi.ome.tif",
        image,
        metadata={"axes": "YX", "PhysicalSizeX": 1.0, "PhysicalSizeY": 1.0},
        ome=True,
    )

    manifest = {
        "run_name": "synthetic_xenium",
        "images": {"morphology_focus": "morphology_focus"},
        "xenium_explorer_files": {
            "cells_zarr": "cells.zarr.zip",
            "cell_feature_matrix_zarr": "cell_feature_matrix.zarr.zip",
            "transcripts_zarr": "transcripts.zarr.zip",
            "analysis_zarr": "analysis.zarr.zip",
            "analysis_summary": "analysis_summary.html",
        },
    }
    (bundle_dir / "experiment.xenium").write_text(json.dumps(manifest))
    (bundle_dir / "analysis_summary.html").write_text("<html></html>")
    (bundle_dir / "gene_panel.json").write_text(json.dumps({"panel_name": "synthetic"}))

    cells_store = ZipStore(str(bundle_dir / "cells.zarr.zip"), mode="w")
    with cells_store:
        cells = zarr.open_group(store=cells_store, mode="w")
        cells.create_array("cell_id", data=np.array([[1, 1], [2, 1]], dtype=np.uint32))
        cells.create_array(
            "cell_summary",
            data=np.array(
                [
                    [1.0, 2.0, 10.0, 1.0, 2.0, 4.0, 0.0, 1.0],
                    [5.0, 6.0, 20.0, 5.0, 6.0, 8.0, 0.0, 1.0],
                ],
                dtype=np.float64,
            ),
        )
        masks = cells.create_group("masks")
        masks.create_array("homogeneous_transform", data=np.eye(4, dtype=np.float32))
        polygon_sets = cells.create_group("polygon_sets")
        cell_polygons = polygon_sets.create_group("1")
        cell_polygons.create_array("cell_index", data=np.array([0, 1], dtype=np.uint32))
        cell_polygons.create_array("method", data=np.array([0, 0], dtype=np.uint32))
        cell_polygons.create_array("num_vertices", data=np.array([0, 0], dtype=np.int32))
        cell_polygons.create_array("vertices", data=np.empty((0, 2), dtype=np.float32))

    cfm_store = ZipStore(str(bundle_dir / "cell_feature_matrix.zarr.zip"), mode="w")
    with cfm_store:
        cfm = zarr.open_group(store=cfm_store, mode="w")
        group = cfm.create_group("cell_features")
        group.create_array("cell_id", data=np.array([[1, 1], [2, 1]], dtype=np.uint32))
        csc = group.create_group("csc")
        csc.create_array("data", data=np.array([3, 5, 7], dtype=np.uint32))
        csc.create_array("indices", data=np.array([0, 1, 1], dtype=np.uint16))
        csc.create_array("indptr", data=np.array([0, 1, 3], dtype=np.uint32))
        group.attrs["feature_keys"] = ["GeneA", "GeneB"]
        group.attrs["feature_ids"] = ["ENSGA", "ENSGB"]
        group.attrs["feature_types"] = ["Gene Expression", "Gene Expression"]

    transcripts_store = ZipStore(str(bundle_dir / "transcripts.zarr.zip"), mode="w")
    with transcripts_store:
        transcripts = zarr.open_group(store=transcripts_store, mode="w")
        transcripts.attrs["gene_names"] = ["GeneA", "GeneB"]
        grids = transcripts.create_group("grids")
        grids.attrs["number_levels"] = 1
        grid0 = grids.create_group("0")
        tile = grid0.create_group("0,0")
        tile.create_array(
            "location",
            data=np.array(
                [
                    [1.0, 2.0, 0.0],
                    [5.0, 6.0, 0.0],
                ],
                dtype=np.float32,
            ),
        )
        tile.create_array("gene_identity", data=np.array([[0], [1]], dtype=np.uint16))
        tile.create_array("quality_score", data=np.array([35.0, 42.0], dtype=np.float32))
        tile.create_array("valid", data=np.array([1, 1], dtype=np.uint8))

    analysis_store = ZipStore(str(bundle_dir / "analysis.zarr.zip"), mode="w")
    with analysis_store:
        analysis = zarr.open_group(store=analysis_store, mode="w")
        analysis.create_group("cell_groups")

    return bundle_dir


def test_end_to_end_bundle_loading(tmp_path: Path):
    bundle_dir = _build_fixture_bundle(tmp_path / "bundle")
    bundle = load_bundle_paths(bundle_dir)
    image = load_morphology_image(bundle.morphology_focus_dir / "ch0000_dapi.ome.tif", max_dim=64)
    cells, transform = load_cells_table(bundle)
    registration = build_registration(transform, image)
    registered_cells = register_cells(cells, registration)
    matrix, var = load_cell_feature_matrix(bundle)
    transcripts = load_transcript_preview(bundle, max_points=100)
    adata = build_h5ad(bundle, image, registration)

    assert matrix.shape == (2, 2)
    assert list(var.index) == ["GeneA", "GeneB"]
    assert len(transcripts.frame) == 2
    assert np.allclose(registered_cells[["x_px", "y_px"]].to_numpy(), [[1.0, 2.0], [5.0, 6.0]])
    assert adata.shape == (2, 2)
    assert "spatial" in adata.obsm
