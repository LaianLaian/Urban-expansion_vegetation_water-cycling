#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
City–subregion extraction for precipitation P only; P-related outputs match aggregate_etp_to_city_scale.py.

Loads W1/W2/V with the same lon/lat handling and masks as the main script; does not read ET / ET_vegenon NetCDF.
P is aligned to the W1 grid via try_load_annual_first_match → load_annual_field_aligned →
align_annual_field_to_w1_grid, then aggregated.

Outputs (same paths/column names as the main script; existing P-only files are removed first to avoid duplicate
rows when re-running in append mode):

  {out_dir}/panel_city_year_region_P.csv
  {out_dir}/city_year_overall_P.csv
  {out_dir}/city_region_climatology_P_2000_2021_mean.csv
  {out_dir}/city_attributes_P.csv
  {out_dir}/by_city/<city>_region_climatology_P_mean.csv
  {out_dir}/by_city/<city>_city_attributes_P.csv

Does not modify main tables (e.g. panel_city_year_region.csv / city_year_overall.csv). To refresh P columns there,
re-run the full aggregate_etp_to_city_scale.py or merge on (city_id, year, region).

Usage (defaults: project root = parent of this ``code/`` directory):

  python aggregate_p_only_to_city_scale.py --out-dir "../city_etp_outputs"
  python aggregate_p_only_to_city_scale.py --precip-nc "../MSWEP_annual_precip_2000_2021.nc"
"""
from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.ops import transform as shp_transform

import aggregate_etp_to_city_scale as etp

PROJECT_ROOT = Path(__file__).resolve().parent.parent

try:
    from tqdm import tqdm as _tqdm_raw

    def tqdm(iterable, **kwargs):
        # tqdm often disables the bar when stderr is not a TTY (IDE / redirection)
        if kwargs.get("disable") is None:
            if not sys.stderr.isatty():
                kwargs["disable"] = False
        return _tqdm_raw(iterable, **kwargs)

    _HAVE_TQDM = True
except ImportError:
    _HAVE_TQDM = False

    def tqdm(iterable, *, total=None, desc="progress", **kwargs):
        """Simple numeric progress when tqdm is not installed."""
        n = 0
        if total is None:
            try:
                total = len(iterable)
            except TypeError:
                total = None
        for item in iterable:
            n += 1
            if total and total > 0:
                step = max(1, total // 100)
                if n == 1 or n % step == 0 or n == total:
                    print(f"\r{desc}: {n}/{total}", end="", flush=True)
            elif n == 1 or (n % 200 == 0):
                print(f"\r{desc}: processed {n} …", end="", flush=True)
            yield item
        if n:
            print(f"\n{desc}: done, {n} items.", flush=True)


def _compute_bundle_one_city_p(
    geom_row: pd.Series,
    *,
    city_id_col: str,
    w1,
    w2,
    v,
    lon_edges: np.ndarray,
    lat_edges: np.ndarray,
    weight_grid: np.ndarray,
    area_threshold: float,
    mask_fb: float | None,
    extra_dict: dict,
    years_ag: np.ndarray,
    year_early_end: int,
    year_late_start: int,
) -> dict:
    """
    One city: zonal panel, overall union, P climatology row, attributes slice; built for threaded execution.
    Each worker constructs its own pyproj Transformer (not safe to share across threads).
    """
    transformer_ag = etp._transformer_ll_to_cea()
    rec_p = etp.records_for_city_region_panel(
        geom_row,
        city_id_col,
        w1,
        w2,
        v,
        lon_edges,
        lat_edges,
        weight_grid,
        area_threshold,
        mask_fb,
        extra_dict,
        transformer_ag,
        years_ag,
    )
    df_p = pd.DataFrame.from_records(rec_p)
    rec_o = etp.records_for_city_overall_union(
        geom_row,
        city_id_col,
        w1,
        w2,
        v,
        lon_edges,
        lat_edges,
        weight_grid,
        area_threshold,
        mask_fb,
        extra_dict,
        transformer_ag,
        years_ag,
    )
    df_o = pd.DataFrame.from_records(rec_o)
    if "P" not in df_p.columns:
        raise RuntimeError("Internal error: panel has no column P.")
    _base3 = ["city_id", "year", "region"]
    _by = ["city_id", "year"]
    cid = geom_row[city_id_col]
    safe_nm = etp.safe_city_filename(cid)
    clim_p_row = etp.panel_to_region_climatology(df_p, ["P"])
    attr_row = etp.compute_city_attributes_one(
        geom_row,
        df_o,
        city_id_col,
        year_early_end,
        year_late_start,
        transformer_ag,
    )
    attrs_p_part = etp.compute_city_attributes_p(
        df_o,
        city_id_col,
        year_early_end=year_early_end,
        year_late_start=year_late_start,
    )
    attrs_out: pd.DataFrame | None = None
    if attrs_p_part is not None and len(attrs_p_part):
        attrs_out = attrs_p_part.copy()
        if "Expansion" in attr_row:
            attrs_out["Expansion"] = attr_row["Expansion"]
    return {
        "panel_p": df_p[_base3 + ["P"]].copy(),
        "overall_p": df_o[_by + ["P"]].copy(),
        "clim_p_row": clim_p_row.copy(),
        "safe_nm": safe_nm,
        "attrs_p_part": attrs_out,
    }


def _clear_previous_p_outputs(out_dir: Path) -> None:
    """Remove P-specific outputs this script will rewrite, avoiding duplicated rows under append mode."""
    paths = [
        out_dir / "panel_city_year_region_P.csv",
        out_dir / "city_year_overall_P.csv",
        out_dir / "city_region_climatology_P_2000_2021_mean.csv",
        out_dir / "city_attributes_P.csv",
    ]
    for p in paths:
        if p.exists():
            p.unlink()
    by_dir = out_dir / "by_city"
    if by_dir.is_dir():
        for pat in ("*_region_climatology_P_mean.csv", "*_city_attributes_P.csv"):
            for p in by_dir.glob(pat):
                p.unlink()


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Aggregate precipitation P to city/subregion scale (same P outputs as aggregate_etp)."
    )
    ap.add_argument(
        "--w1",
        type=Path,
        default=PROJECT_ROOT / "ET_over_P_annual_2000_2021.nc",
    )
    ap.add_argument(
        "--w2",
        type=Path,
        default=PROJECT_ROOT / "ET_vegenon_over_P_annual_2000_2021.nc",
    )
    ap.add_argument("--var-w1", type=str, default="ET_over_P")
    ap.add_argument("--var-w2", type=str, default="ET_vegenon_over_P")
    ap.add_argument(
        "--precip-nc",
        type=Path,
        default=PROJECT_ROOT / "MSWEP_annual_precip_2000_2021.nc",
    )
    ap.add_argument("--var-precip", type=str, default="P")
    ap.add_argument("--city-id-col", type=str, default="auto")
    ap.add_argument(
        "--shp-dir",
        type=Path,
        default=PROJECT_ROOT / "filtered_outputs",
    )
    ap.add_argument("--gub-2000", type=str, default="GUB_Global_2000_filtered.shp")
    ap.add_argument("--gub-2020", type=str, default="GUB_Global_2020_filtered.shp")
    ap.add_argument("--suburban-2020", type=str, default="GUB_suburban_30km_2020.shp")
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=PROJECT_ROOT / "city_etp_outputs",
    )
    ap.add_argument("--area-threshold", type=float, default=0.5)
    ap.add_argument("--mask-fallback-threshold", type=float, default=0.0)
    ap.add_argument("--interp-res-deg", type=float, default=None)
    ap.add_argument("--year-early-end", type=int, default=2010)
    ap.add_argument("--year-late-start", type=int, default=2010)
    ap.add_argument(
        "--no-clear",
        action="store_true",
        help="Do not delete existing P CSVs at start (caution: re-runs append duplicate rows).",
    )
    ap.add_argument(
        "--workers",
        type=int,
        default=0,
        help=(
            "Thread count per city; 0 = auto min(12, max(2, CPUs)); "
            "1 = single-threaded (debugging); >1 uses a thread pool (shared in-memory grids)."
        ),
    )
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    if not args.no_clear:
        _clear_previous_p_outputs(args.out_dir)

    print("Loading W1/W2/V (same as main script)…")
    w1, w2, v = etp.load_w1_w2_v(args.w1, args.w2, args.var_w1, args.var_w2)
    lon = np.asarray(w1["lon"].values, dtype=np.float64)
    lon_axis = etp.infer_lon_axis(lon)
    if lon_axis == "-180..180":
        lon = etp.lon_to_0_360(lon)
        w1 = w1.assign_coords(lon=lon)
        w2 = w2.assign_coords(lon=lon)
        v = v.assign_coords(lon=lon)
    w1 = w1.sortby("lon").sortby("lat")
    w2 = w2.sortby("lon").sortby("lat")
    v = v.sortby("lon").sortby("lat")

    print("Loading precipitation P and aligning to W1 grid…")
    p_vars = [
        args.var_precip,
        "annual_pricip",
        "annual_precip",
        "precipitation",
        "P",
    ]
    da_p = etp.try_load_annual_first_match(args.precip_nc, p_vars, w1)
    if da_p is None:
        raise SystemExit(
            f"Could not load P from {args.precip_nc} (tried variables: {', '.join(p_vars)})."
        )
    extra_fields: dict[str, object] = {"P": da_p}
    print("  Loaded P:", args.precip_nc.name)

    if args.interp_res_deg is not None:
        ddeg = float(args.interp_res_deg)
        print(f"Spatial interpolation: bilinear to ~{ddeg}° …")
        w1 = etp.resample_lat_lon_linear_deg(w1, ddeg)
        w2 = etp.resample_lat_lon_linear_deg(w2, ddeg)
        v = w1 - w2
        v.name = "V"
        extra_fields["P"] = etp.resample_lat_lon_linear_deg(da_p, ddeg)

    lon = np.asarray(w1["lon"].values, dtype=np.float64)
    lat = np.asarray(w1["lat"].values, dtype=np.float64)
    dlon = float(np.nanmean(np.diff(lon)))
    dlat = float(np.nanmean(np.diff(lat)))
    lon_edges = np.concatenate(
        [[lon[0] - dlon / 2], (lon[:-1] + lon[1:]) / 2, [lon[-1] + dlon / 2]]
    )
    lat_edges = np.concatenate(
        [[lat[0] - dlat / 2], (lat[:-1] + lat[1:]) / 2, [lat[-1] + dlat / 2]]
    )
    if np.nanmin(lon) >= 0 and np.nanmax(lon) <= 360.0:
        lon_edges[0] = max(0.0, float(lon_edges[0]))
        lon_edges[-1] = min(360.0, float(lon_edges[-1]))
    else:
        lon_edges = (lon_edges + 360.0) % 360.0

    weight_grid = etp.cell_area_weights(lat, lon)
    p_da = extra_fields["P"]
    if p_da.shape != w1.shape:
        raise ValueError(f"P shape {p_da.shape} does not match W1 {w1.shape}")

    def resolve_shp(directory: Path, filename: str, fallbacks: tuple[str, ...]) -> Path:
        candidates = (filename,) + fallbacks
        for name in candidates:
            p = directory / name
            if p.exists():
                return p
        tried = ", ".join(candidates)
        raise FileNotFoundError(f"Under {directory} none of these exist: {tried}")

    p2000 = resolve_shp(
        args.shp_dir,
        args.gub_2000,
        ("GUB_2000.shp", "GUB_Global_2000_filtered.shp"),
    )
    p2020 = resolve_shp(
        args.shp_dir,
        args.gub_2020,
        ("GUB_2020.shp", "GUB_Global_2020_filtered.shp"),
    )
    psub = resolve_shp(
        args.shp_dir,
        args.suburban_2020,
        ("suburban_2020.shp", "GUB_suburban_30km_2020.shp"),
    )
    print("Using shapefiles:", p2000.name, p2020.name, psub.name)

    id_kw = None if args.city_id_col.strip().lower() == "auto" else args.city_id_col
    g0 = etp.add_unified_city_id(etp.prepare_geoms_wgs84(gpd.read_file(p2000)), id_kw)
    g2 = etp.add_unified_city_id(etp.prepare_geoms_wgs84(gpd.read_file(p2020)), id_kw)
    gs = etp.add_unified_city_id(etp.prepare_geoms_wgs84(gpd.read_file(psub)), id_kw)

    def shift_lon_geom(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        gdf = gdf.copy()

        def shift(g):
            if g is None or g.is_empty:
                return g
            return shp_transform(lambda x, y: ((x + 360) % 360, y), g)

        gdf["geometry"] = gdf.geometry.apply(shift)
        return gdf

    g0 = shift_lon_geom(g0)
    g2 = shift_lon_geom(g2)
    gs = shift_lon_geom(gs)

    print(
        "Building Core / Expansion / Peri (no progress bar; many cities may take minutes)…",
        flush=True,
    )
    city_geoms = etp.build_core_expansion_peri(g0, g2, gs, etp.CITY_ID_COL)
    d2u = etp.dissolve_by_city(g2, etp.CITY_ID_COL)
    d2u = d2u.rename(columns={"geometry": "geometry_2020_full"})
    city_geoms = city_geoms.merge(
        d2u[[etp.CITY_ID_COL, "geometry_2020_full"]], on=etp.CITY_ID_COL, how="left"
    )

    CITY_ID_COL = etp.CITY_ID_COL
    extra_dict = {"P": p_da}
    years_ag = etp._panel_years_from_w1(w1)

    by_dir = args.out_dir / "by_city"
    by_dir.mkdir(parents=True, exist_ok=True)
    p_p_path = args.out_dir / "panel_city_year_region_P.csv"
    o_p_path = args.out_dir / "city_year_overall_P.csv"
    clim_p_path = args.out_dir / "city_region_climatology_P_2000_2021_mean.csv"
    attrs_p_path = args.out_dir / "city_attributes_P.csv"

    fp_ponly = True
    fo_ponly = True
    fclim_p = True
    fattrs_p = True

    mask_fb: float | None = (
        None if args.mask_fallback_threshold < 0 else float(args.mask_fallback_threshold)
    )

    n_city = len(city_geoms)
    nw = int(args.workers)
    if nw <= 0:
        nw = min(12, max(2, (os.cpu_count() or 4)))
    else:
        nw = max(1, nw)
    if not _HAVE_TQDM:
        print(
            "Tip: install tqdm for a clearer progress bar: pip install tqdm",
            flush=True,
        )
    print(
        f"Aggregating P per city ({n_city} cities, {nw} threads; P CSVs only)…",
        flush=True,
    )

    city_rows = [r.copy() for _, r in city_geoms.iterrows()]
    _fn = partial(
        _compute_bundle_one_city_p,
        city_id_col=CITY_ID_COL,
        w1=w1,
        w2=w2,
        v=v,
        lon_edges=lon_edges,
        lat_edges=lat_edges,
        weight_grid=weight_grid,
        area_threshold=args.area_threshold,
        mask_fb=mask_fb,
        extra_dict=extra_dict,
        years_ag=years_ag,
        year_early_end=args.year_early_end,
        year_late_start=args.year_late_start,
    )
    with ThreadPoolExecutor(max_workers=nw) as ex:
        bundles = list(
            tqdm(
                ex.map(_fn, city_rows),
                total=n_city,
                desc="cities",
                file=sys.stderr,
            )
        )

    for pack in bundles:
        etp._append_df_csv(p_p_path, pack["panel_p"], first=fp_ponly)
        fp_ponly = False
        etp._append_df_csv(o_p_path, pack["overall_p"], first=fo_ponly)
        fo_ponly = False
        etp._append_df_csv(clim_p_path, pack["clim_p_row"], first=fclim_p)
        fclim_p = False
        pack["clim_p_row"].to_csv(
            by_dir / f"{pack['safe_nm']}_region_climatology_P_mean.csv",
            index=False,
        )
        attrs_p_part = pack["attrs_p_part"]
        if attrs_p_part is not None and len(attrs_p_part):
            etp._append_df_csv(attrs_p_path, attrs_p_part, first=fattrs_p)
            fattrs_p = False
            attrs_p_part.to_csv(
                by_dir / f"{pack['safe_nm']}_city_attributes_P.csv",
                index=False,
            )

    print("Written:", p_p_path)
    print("Written:", o_p_path)
    print("Written:", clim_p_path)
    print("Written:", attrs_p_path)
    print("Per-city P climatology / attributes:", by_dir)
    meta = args.out_dir / "RUN_METADATA_P_only.txt"
    meta.write_text(
        "aggregate_p_only_to_city_scale.py: recompute precipitation P only; filenames match aggregate_etp P branch.\n"
        f"W1: {args.w1}\nW2: {args.w2}\nP: {args.precip_nc}\n"
        f"interp_res_deg: {args.interp_res_deg}\n"
        f"workers: {nw}\n",
        encoding="utf-8",
    )
    print("Metadata:", meta)
    print("Done.")


if __name__ == "__main__":
    main()
