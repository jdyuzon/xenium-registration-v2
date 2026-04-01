from __future__ import annotations

import argparse
import re
from pathlib import Path

import anndata as ad
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from xenium_app.io import load_bundle_paths
from xenium_app.xenium import (
    build_h5ad,
    build_registration,
    load_all_channel_images,
    load_cells_table,
    load_morphology_image,
    load_morphology_source_from_bundle,
)


def parse_cell_type_csv(path: Path) -> pd.DataFrame:
    table = pd.read_csv(path)
    if table.shape[1] < 2:
        raise ValueError(f"{path} must have at least two columns, including cell_id and one label column.")

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


def build_obs_name_map(columns: list[str]) -> dict[str, str]:
    rename_map: dict[str, str] = {}
    used: set[str] = set()
    for column in columns:
        base = sanitize_metadata_name(column)
        candidate = base
        suffix = 2
        while candidate in used:
            candidate = f"{base}_{suffix}"
            suffix += 1
        rename_map[column] = candidate
        used.add(candidate)
    return rename_map


def pick_dapi_channel_index(channel_names: list[str]) -> int:
    return next((idx for idx, name in enumerate(channel_names) if "dapi" in name.lower()), 0)


def load_manifest(path: Path) -> pd.DataFrame:
    manifest = pd.read_csv(path)
    required = {"bundle_path", "sample_id"}
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError(f"Manifest missing required columns: {sorted(missing)}")

    manifest = manifest.copy()
    if "project_id" not in manifest.columns:
        manifest["project_id"] = manifest["sample_id"]
    if "label_csv" not in manifest.columns:
        manifest["label_csv"] = ""
    return manifest


def load_bundle_adata(row: pd.Series) -> ad.AnnData:
    bundle = load_bundle_paths(row["bundle_path"])
    all_images = load_all_channel_images(bundle, max_dim=2048)
    dapi_index = pick_dapi_channel_index([img.channel_name for img in all_images])
    image = all_images[dapi_index]
    extra_images = [img for i, img in enumerate(all_images) if i != dapi_index]
    _, transform = load_cells_table(bundle)
    registration = build_registration(transform, image)

    extra_obs = pd.DataFrame(index=[])
    label_csv = str(row.get("label_csv", "")).strip()
    if label_csv:
        extra_obs = parse_cell_type_csv(Path(label_csv))

    probe_obs = pd.DataFrame(
        {
            "sample_id": [],
            "project_id": [],
            "bundle_path": [],
        }
    )
    obs_name_map = build_obs_name_map(
        list(
            pd.DataFrame(
                columns=[
                    "cell_centroid_x",
                    "cell_centroid_y",
                    "cell_area",
                    "nucleus_centroid_x",
                    "nucleus_centroid_y",
                    "nucleus_area",
                    "z_level",
                    "nucleus_count",
                    "x_px",
                    "y_px",
                    "image_channel",
                ]
            )
            .join(extra_obs, how="left")
            .join(probe_obs, how="left")
            .columns
        )
    )

    adata = build_h5ad(bundle, image, registration, extra_obs=extra_obs, obs_name_map=obs_name_map, extra_images=extra_images)
    adata.obs["sample_id"] = row["sample_id"]
    adata.obs["project_id"] = row["project_id"]
    adata.obs["bundle_path"] = str(Path(row["bundle_path"]).expanduser())
    adata.obs_names = [f"{row['sample_id']}::{cell_id}" for cell_id in adata.obs_names]
    adata.uns["dataset_metadata"] = {
        "sample_id": row["sample_id"],
        "project_id": row["project_id"],
        "bundle_path": str(Path(row["bundle_path"]).expanduser()),
        "label_csv": label_csv,
    }
    return adata


def assign_splits(obs: pd.DataFrame, group_col: str, val_frac: float, test_frac: float, seed: int) -> pd.Series:
    groups = obs[group_col].astype(str).to_numpy()
    indices = obs.index.to_numpy()
    dummy = obs.index.to_series().reset_index(drop=True)

    if val_frac + test_frac >= 1.0:
        raise ValueError("val_frac + test_frac must be less than 1.")

    splitter = GroupShuffleSplit(n_splits=1, test_size=test_frac, random_state=seed)
    train_val_idx, test_idx = next(splitter.split(dummy, groups=groups))

    train_val_obs = obs.iloc[train_val_idx]
    remaining_val_frac = val_frac / (1.0 - test_frac) if (1.0 - test_frac) > 0 else 0.0
    if remaining_val_frac <= 0:
        train_idx = train_val_idx
        val_idx = []
    else:
        splitter_val = GroupShuffleSplit(n_splits=1, test_size=remaining_val_frac, random_state=seed + 1)
        rel_train_idx, rel_val_idx = next(
            splitter_val.split(train_val_obs.index.to_series().reset_index(drop=True), groups=train_val_obs[group_col].astype(str).to_numpy())
        )
        train_idx = train_val_obs.iloc[rel_train_idx].index
        val_idx = train_val_obs.iloc[rel_val_idx].index

    split = pd.Series("train", index=obs.index, dtype="object")
    split.loc[obs.iloc[test_idx].index] = "test"
    split.loc[val_idx] = "val"
    split.loc[train_idx] = "train"
    return split


def main() -> None:
    parser = argparse.ArgumentParser(description="Build merged Xenium ML dataset and grouped train/val/test splits.")
    parser.add_argument("--manifest", required=True, help="CSV with bundle_path,sample_id and optional project_id,label_csv.")
    parser.add_argument("--output-dir", required=True, help="Directory to write outputs.")
    parser.add_argument("--group-col", default="sample_id", choices=["sample_id", "project_id"], help="Group splits by sample or project.")
    parser.add_argument("--val-frac", type=float, default=0.2, help="Validation fraction by group.")
    parser.add_argument("--test-frac", type=float, default=0.2, help="Test fraction by group.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for grouped split.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest(Path(args.manifest).expanduser().resolve())
    adatas = [load_bundle_adata(row) for _, row in manifest.iterrows()]
    merged = ad.concat(adatas, axis=0, join="outer", merge="same", label="source_dataset", keys=manifest["sample_id"].tolist())
    merged.obs["split"] = assign_splits(merged.obs, group_col=args.group_col, val_frac=args.val_frac, test_frac=args.test_frac, seed=args.seed)

    merged_path = output_dir / "merged_ml_dataset.h5ad"
    merged.write_h5ad(merged_path)

    merged.obs.to_csv(output_dir / "merged_obs_with_splits.csv")
    manifest.to_csv(output_dir / "input_manifest_resolved.csv", index=False)

    split_summary = (
        merged.obs.groupby(["split", args.group_col])
        .size()
        .reset_index(name="cell_count")
        .sort_values(["split", args.group_col])
    )
    split_summary.to_csv(output_dir / "split_summary_by_group.csv", index=False)

    class_col = "cell_type" if "cell_type" in merged.obs.columns else None
    if class_col:
        class_summary = (
            merged.obs.groupby(["split", class_col])
            .size()
            .reset_index(name="cell_count")
            .sort_values(["split", class_col])
        )
        class_summary.to_csv(output_dir / "split_summary_by_class.csv", index=False)

    print(f"Wrote merged dataset to {merged_path}")
    print(f"Split group column: {args.group_col}")
    print(merged.obs["split"].value_counts().to_string())


if __name__ == "__main__":
    main()
