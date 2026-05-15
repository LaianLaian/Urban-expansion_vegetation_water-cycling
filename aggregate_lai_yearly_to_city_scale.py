#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Monthly LAI gridded NetCDF (default variable ``lai``) → yearly means → longitude/latitude aligned with shapefiles →
cos(lat)-weighted zonal aggregation over core / expansion / peri and overall union; writes Excel.

Outputs mirror aggregate_etp_to_city_scale.py: **master CSV appended per city**
(``panel_city_year_region_LAI.csv``, ``city_year_overall_LAI.csv``), two CSVs per city under ``by_city/``,
plus Excel and trend CSVs. Within a chosen year interval (default 2000–2021), Theil–Sen slope and multi-year mean
of LAI versus time are computed.

Same conventions as aggregate_etp_to_city_scale.py:
- If grid longitude is -180..180, assign lon 0..360 and shift shapefile geometries to the same domain.
- Reorder lon/lat to ascending on the **annual** field only, to avoid duplicating the full monthly cube (OOM).
- Cell intersection–fraction thresholding and cos(lat) weights match the ET/P script.

If a city raises during processing, two NaN CSVs and a ``*_FAILED.txt`` are still written and other cities continue.
If U2000∩U2020 has no area (empty core geometry), region=core LAI is NaN by definition.
For optional imputation of core from city-wide LAI when core is empty, use
``--impute-core-lai-from-overall-when-core-empty`` (see impute_core_lai_from_overall_if_core_geom_empty).

Dependencies: numpy, pandas, xarray, geopandas, shapely, pyproj, scipy; openpyxl for Excel.
Optional: tqdm

Filename convention: ``*_YYYY.nc`` (e.g. ``lai_monthly_0.05_2000.nc``) — ``YYYY`` is the calendar year of the
monthly series inside the file; the script rewrites that file's time coordinate to YYYY-01 …

Example:
  python aggregate_lai_yearly_to_city_scale.py ^
    --lai-dir "../LAImon/LAI" ^
    --out-dir "../city_lai_outputs" ^
    --out-xlsx lai_city_yearly.xlsx
"""
from __future__ import annotations

import argparse
import re
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
from shapely.geometry import Polygon
from shapely.ops import transform as shp_transform, unary_union

try:
    from tqdm import tqdm
except ImportError:

    def tqdm(x, **kwargs):
        return x

import aggregate_etp_to_city_scale as etp

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# e.g. lai_monthly_0.05_2000.nc → trailing 2000 means monthly data for that calendar year
_YEAR_BEFORE_NC_RE = re.compile(r"_(\d{4})\s*\.nc$", re.IGNORECASE)


def parse_year_from_lai_filename(path: Path) -> int | None:
    """Parse year from trailing ``_YYYY.nc`` (prefix such as resolution is ignored)."""
    m = _YEAR_BEFORE_NC_RE.search(path.name)
    if not m:
        return None
    y = int(m.group(1))
    if 1000 <= y <= 3000:
        return y
    return None


def _assign_monthly_time_from_filename_if_known(
    da: xr.DataArray, path: Path
) -> xr.DataArray:
    """
    When the filename contains ``_YYYY.nc``, set ``time`` to monthly starts YYYY-01 …
    (same length as the current ``time``, usually 12) for concatenation and annual means.
    """
    da = _rename_time_dim(_rename_lat_lon_dims(da))
    yr = parse_year_from_lai_filename(path)
    if yr is None or "time" not in da.dims:
        return da
    nt = int(da.sizes["time"])
    if nt < 1:
        return da
    times = pd.date_range(f"{yr}-01-01", periods=nt, freq="MS")
    return da.assign_coords(time=("time", times))


def _rename_lat_lon_dims(da: xr.DataArray) -> xr.DataArray:
    m: dict[str, str] = {}
    for old, new in (
        ("latitude", "lat"),
        ("Latitude", "lat"),
        ("LAT", "lat"),
        ("longitude", "lon"),
        ("Longitude", "lon"),
        ("LON", "lon"),
    ):
        if old in da.dims and new not in da.dims:
            m[old] = new
    return da.rename(m) if m else da


def _rename_time_dim(da: xr.DataArray) -> xr.DataArray:
    for old, new in (("Time", "time"), ("MONTH", "time"), ("month", "time")):
        if old in da.dims and "time" not in da.dims:
            da = da.rename({old: new})
    return da


def _reorder_lon_lat_ascending(da: xr.DataArray) -> xr.DataArray:
    """
    Sort pixel order by ascending lon/lat without xarray.sortby (avoids costly deep copies / OOM).
    Implemented locally — no separate reorder helper in aggregate_etp_to_city_scale.
    """
    da = etp._normalize_to_time_lat_lon(da)
    lon = np.asarray(da["lon"].values, dtype=np.float64)
    lat = np.asarray(da["lat"].values, dtype=np.float64)
    if lon.size >= 2:
        lon_idx = np.argsort(lon)
        if not np.array_equal(lon_idx, np.arange(lon.size)):
            da = da.isel(lon=lon_idx)
            lat = np.asarray(da["lat"].values, dtype=np.float64)
    if lat.size >= 2:
        lat_idx = np.argsort(lat)
        if not np.array_equal(lat_idx, np.arange(lat.size)):
            da = da.isel(lat=lat_idx)
    return etp._normalize_to_time_lat_lon(da)


def _load_one_nc_var(path: Path, var: str) -> xr.DataArray:
    with xr.open_dataset(path, decode_times=True) as ds:
        if var not in ds:
            raise KeyError(
                f"{path.name}: no variable '{var}'; present: {list(ds.data_vars)}"
            )
        return ds[var].load()


def load_lai_monthly_stack(lai_dir: Path, var: str) -> xr.DataArray:
    """
    Load all ``.nc`` in a directory into a monthly ``time``, ``lat``, ``lon`` cube.

    Convention: for ``lai_monthly_0.05_2000.nc``, the trailing ``_2000`` marks **calendar year**
    monthly stacks; ``time`` is set to monthly starts then files are sorted by year and concatenated along ``time``.
    Files without parseable years keep NetCDF-native ``time`` and sort after dated files.
    """
    paths = sorted(lai_dir.glob("*.nc"))
    if not paths:
        raise FileNotFoundError(f"No .nc files in directory: {lai_dir}")

    def pack(da: xr.DataArray) -> xr.DataArray:
        da = _rename_time_dim(_rename_lat_lon_dims(da))
        return etp._normalize_to_time_lat_lon(da)

    def _sort_key(p: Path) -> tuple:
        y = parse_year_from_lai_filename(p)
        if y is not None:
            return (0, y, p.name)
        return (1, 0, p.name)

    if len(paths) == 1:
        da = _load_one_nc_var(paths[0], var)
        da = _assign_monthly_time_from_filename_if_known(da, paths[0])
        return pack(da)

    parts: list[xr.DataArray] = []
    for p in sorted(paths, key=_sort_key):
        da = _load_one_nc_var(p, var)
        da = _assign_monthly_time_from_filename_if_known(da, p)
        parts.append(da)

    if all("time" not in p.dims for p in parts):
        parts = [p.expand_dims(time=[i]) for i, p in enumerate(parts)]

    try:
        da = xr.concat(parts, dim="time", coords="minimal", compat="override")
    except Exception as e:
        raise RuntimeError(
            "Concatenating monthly files along time failed (check identical lat/lon grids across files)."
            f" Detail: {e!r}"
        ) from e

    if "time" in da.dims and da.sizes["time"] > 1:
        try:
            da = da.sortby("time")
        except Exception:
            pass
    return pack(da)


def assign_lon_0_360_for_shp(da: xr.DataArray) -> xr.DataArray:
    """
    Shapefile-aligned: if grid is -180..180, rewrite ``lon`` to 0..360 via assign_coords **without reordering pixels**.

    Full lon/lat **isel** reordering is deferred: reordering (~264-month, lat, lon) would duplicate the entire cube (tens
    of GB) and often OOM. Reordering runs after monthly_to_yearly_mean on the yearly field only.
    """
    da = etp._normalize_to_time_lat_lon(da)
    lon = np.asarray(da["lon"].values, dtype=np.float64)
    lon_axis = etp.infer_lon_axis(lon)
    if lon_axis == "-180..180":
        lon = etp.lon_to_0_360(lon)
        da = da.assign_coords(lon=lon)
    return da


def _calendar_year_per_time_index(times: np.ndarray) -> np.ndarray:
    """Calendar year (int32) for each ``time`` index; supports datetime64, cftime, and Timestamp-like."""
    tv = np.asarray(times)
    if tv.size == 0:
        return np.array([], dtype=np.int32)
    if np.issubdtype(tv.dtype, np.datetime64):
        return pd.to_datetime(tv).year.values.astype(np.int32)
    out = np.empty(tv.shape[0], dtype=np.int32)
    for i, t in enumerate(tv):
        if hasattr(t, "year"):
            out[i] = int(t.year)
        else:
            out[i] = int(pd.Timestamp(t).year)
    return out


def monthly_to_yearly_mean(da: xr.DataArray) -> xr.DataArray:
    """
    Mean monthly LAI within each calendar year; output ``time`` is one sample per year (coordinate = Jan 1).

    For coarse global grids, avoid xarray ``groupby/resample.mean`` paths that allocate large temporaries — this uses
    per-year accumulation (sum/count) with peak memory of one (lat, lon) accumulator slab plus monthly slices.
    """
    da = etp._normalize_to_time_lat_lon(da)
    if "time" not in da.dims:
        raise ValueError("Monthly LAI has no time dimension")
    nt = int(da.sizes["time"])
    nlat = int(da.sizes["lat"])
    nlon = int(da.sizes["lon"])
    years_arr = _calendar_year_per_time_index(da["time"].values)
    if years_arr.size != nt:
        raise ValueError("time length mismatches inferred year labels")
    uy = np.unique(years_arr)
    n_out = int(uy.size)
    out_vals = np.empty((n_out, nlat, nlon), dtype=np.float32)
    acc = np.zeros((nlat, nlon), dtype=np.float64)
    cnt = np.zeros((nlat, nlon), dtype=np.int16)

    for yi, y in enumerate(uy):
        acc.fill(0.0)
        cnt.fill(0)
        for ti in range(nt):
            if years_arr[ti] != y:
                continue
            slab = np.asarray(da.isel(time=ti).values, dtype=np.float32)
            m = np.isfinite(slab)
            acc[m] += slab[m].astype(np.float64, copy=False)
            cnt[m] += 1
        out_vals[yi] = np.where(
            cnt > 0, (acc / cnt.astype(np.float64)).astype(np.float32), np.nan
        )

    new_time = pd.to_datetime([f"{int(y)}-01-01" for y in uy])
    out = xr.DataArray(
        out_vals,
        dims=("time", "lat", "lon"),
        coords={
            "time": ("time", new_time),
            "lat": da.coords["lat"],
            "lon": da.coords["lon"],
        },
        name=getattr(da, "name", "lai") or "lai",
    )
    for k, v in da.attrs.items():
        out.attrs[k] = v
    return etp._normalize_to_time_lat_lon(out)


def _years_from_time_coord(da: xr.DataArray) -> np.ndarray:
    tv = da["time"].values
    if np.issubdtype(np.asarray(tv).dtype, np.datetime64):
        return pd.to_datetime(tv).year.values.astype(int)
    # Integer or float-encoded years
    return np.asarray(tv, dtype=int).ravel()


def _nan_lai_panel_records(city_id: object, years: np.ndarray, id_col: str) -> list[dict]:
    """Placeholder rows when rasterization fails — still emits two CSVs per city."""
    rows: list[dict] = []
    for yv in years:
        for reg in ("core", "expansion", "peri"):
            rows.append({id_col: city_id, "year": int(yv), "region": reg, "LAI": float("nan")})
    return rows


def _nan_lai_overall_records(city_id: object, years: np.ndarray, id_col: str) -> list[dict]:
    return [{id_col: city_id, "year": int(yv), "LAI": float("nan")} for yv in years]


def records_lai_region_panel(
    row: pd.Series,
    id_col: str,
    lai_y: xr.DataArray,
    lon_edges: np.ndarray,
    lat_edges: np.ndarray,
    weight_grid: np.ndarray,
    threshold: float,
    mask_fallback_threshold: float | None,
    transformer,
    years: np.ndarray,
) -> list[dict]:
    cid = row[id_col]
    regions = ("core", "expansion", "peri")
    geom_cols = ("geometry_core", "geometry_expansion", "geometry_peri")
    records: list[dict] = []
    for reg, gcol in zip(regions, geom_cols):
        poly = row[gcol]
        poly = etp.clean_geom(poly)
        mask_blk, sl_r, sl_c = etp.rasterize_fraction_mask_with_fallback(
            poly,
            lon_edges,
            lat_edges,
            transformer,
            threshold,
            mask_fallback_threshold,
        )
        if mask_blk.size == 0 or not np.any(mask_blk):
            for yv in years:
                records.append(
                    {"city_id": cid, "year": int(yv), "region": reg, "LAI": np.nan}
                )
            continue
        w_sub = weight_grid[sl_r, sl_c]
        for yi, yv in enumerate(years):
            a = lai_y.isel(time=yi).values[sl_r, sl_c]
            records.append(
                {
                    "city_id": cid,
                    "year": int(yv),
                    "region": reg,
                    "LAI": etp.weighted_mean_2d(a, w_sub, mask_blk),
                }
            )
    return records


def records_lai_city_overall(
    row: pd.Series,
    id_col: str,
    lai_y: xr.DataArray,
    lon_edges: np.ndarray,
    lat_edges: np.ndarray,
    weight_grid: np.ndarray,
    threshold: float,
    mask_fallback_threshold: float | None,
    transformer,
    years: np.ndarray,
) -> list[dict]:
    cid = row[id_col]
    parts = []
    for gcol in ("geometry_core", "geometry_expansion", "geometry_peri"):
        p = row[gcol]
        if p is None or getattr(p, "is_empty", True):
            continue
        p = etp.clean_geom(p)
        if p.is_empty:
            continue
        parts.append(p)
    if not parts:
        return [
            {"city_id": cid, "year": int(yv), "LAI": np.nan} for yv in years
        ]
    poly_u = unary_union(parts)
    mask_blk, sl_r, sl_c = etp.rasterize_fraction_mask_with_fallback(
        poly_u,
        lon_edges,
        lat_edges,
        transformer,
        threshold,
        mask_fallback_threshold,
    )
    w_sub = weight_grid[sl_r, sl_c]
    if mask_blk.size == 0 or not np.any(mask_blk):
        return [
            {"city_id": cid, "year": int(yv), "LAI": np.nan} for yv in years
        ]
    out = []
    for yi, yv in enumerate(years):
        a = lai_y.isel(time=yi).values[sl_r, sl_c]
        out.append(
            {
                "city_id": cid,
                "year": int(yv),
                "LAI": etp.weighted_mean_2d(a, w_sub, mask_blk),
            }
        )
    return out


def impute_core_lai_from_overall_if_core_geom_empty(
    rec_p: list[dict],
    rec_o: list[dict],
    geom_row: pd.Series,
) -> list[dict]:
    """
    When ``geometry_core`` is empty (U2000∩U2020 zero area), copy **city-wide** LAI (core∪expansion∪peri union) into panel
    rows ``region=='core'`` for each year.

    After filling, ``core`` no longer means intersection-only LAI; use only when avoiding all-NaN panels —
    omit for strictly ``stable-built`` analyses (keep NaN or drop affected cities).
    """
    c = etp.clean_geom(geom_row.get("geometry_core"))
    if c is not None and not c.is_empty:
        return rec_p
    overall_by_year = {int(o["year"]): o["LAI"] for o in rec_o}
    for r in rec_p:
        if r.get("region") != "core":
            continue
        yr = int(r["year"])
        if yr in overall_by_year:
            r["LAI"] = overall_by_year[yr]
    return rec_p


def shift_lon_geom(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    gdf = gdf.copy()

    def shift(g):
        if g is None or g.is_empty:
            return g
        return shp_transform(lambda x, y: ((x + 360) % 360, y), g)

    gdf["geometry"] = gdf.geometry.apply(shift)
    return gdf


def resolve_shp(directory: Path, filename: str, fallbacks: tuple[str, ...]) -> Path:
    for name in (filename,) + fallbacks:
        p = directory / name
        if p.exists():
            return p
    tried = ", ".join((filename,) + fallbacks)
    raise FileNotFoundError(f"Under {directory}, none found: {tried}")


def lai_trends_by_group(
    df: pd.DataFrame,
    group_cols: list[str],
    *,
    year_col: str = "year",
    value_col: str = "LAI",
    year_start: int,
    year_end: int,
) -> pd.DataFrame:
    """
    Within each group, take yearly values inside [year_start, year_end] inclusively,
    computing multi-year mean and Theil–Sen slope (units: LAI / year); fewer than 3 finite values → NaN trend.
    """
    rows: list[dict] = []
    for key_tuple, g in df.groupby(group_cols, sort=False):
        if not isinstance(key_tuple, tuple):
            key_tuple = (key_tuple,)
        sub = g[(g[year_col] >= year_start) & (g[year_col] <= year_end)].sort_values(
            year_col
        )
        y = sub[year_col].values.astype(float)
        v = sub[value_col].values.astype(float)
        trend = etp._theil_slope(y, v)
        mean = float(np.nanmean(v)) if len(sub) else float("nan")
        n_valid = int(np.nansum(np.isfinite(v)))
        row = {
            **dict(zip(group_cols, key_tuple)),
            "year_start": year_start,
            "year_end": year_end,
            "LAI_mean": mean,
            "LAI_trend_per_year": trend,
            "n_years_in_window": int(len(sub)),
            "n_valid_LAI": n_valid,
        }
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Monthly LAI → yearly means → city / sub-region zonal aggregation → Excel"
    )
    ap.add_argument(
        "--lai-dir",
        type=Path,
        default=PROJECT_ROOT / "LAImon" / "LAI",
        help="Directory of monthly LAI NetCDF files",
    )
    ap.add_argument("--var-lai", type=str, default="lai", help="LAI variable name in NetCDF")
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
        default=PROJECT_ROOT / "city_lai_outputs",
    )
    ap.add_argument("--out-xlsx", type=str, default="lai_city_yearly_mean.xlsx")
    ap.add_argument("--city-id-col", type=str, default="auto")
    ap.add_argument("--area-threshold", type=float, default=0.5)
    ap.add_argument("--mask-fallback-threshold", type=float, default=0.0)
    ap.add_argument("--interp-res-deg", type=float, default=None)
    ap.add_argument(
        "--impute-core-lai-from-overall-when-core-empty",
        action="store_true",
        help="When geometry_core is empty, fill region=core with city-wide LAI (see docstring).",
    )
    ap.add_argument(
        "--trend-year-start",
        type=int,
        default=2000,
        help="Inclusive lower bound of years for trends and climatologies",
    )
    ap.add_argument(
        "--trend-year-end",
        type=int,
        default=2021,
        help="Inclusive upper bound of years for trends and climatologies",
    )
    args = ap.parse_args()

    try:
        import openpyxl  # noqa: F401
    except ImportError as e:
        raise SystemExit("Excel export requires openpyxl: pip install openpyxl") from e

    args.out_dir.mkdir(parents=True, exist_ok=True)
    xlsx_path = args.out_dir / args.out_xlsx

    print("Reading monthly LAI …")
    lai_m = load_lai_monthly_stack(args.lai_dir, args.var_lai)
    print(
        f"  Monthly cube: time={lai_m.sizes.get('time')} lat={lai_m.sizes.get('lat')} lon={lai_m.sizes.get('lon')}"
    )

    print("Longitude 0..360 for grid (monthly cube not lon/lat reordered to save memory)…")
    lai_m = assign_lon_0_360_for_shp(lai_m)

    print("Computing yearly means …")
    lai_y = monthly_to_yearly_mean(lai_m)
    del lai_m
    lai_y.name = "LAI"
    years = _years_from_time_coord(lai_y)
    print(f"  Years: {len(years)}, range {int(years.min())}–{int(years.max())}")

    print("Annual field: sort lon/lat ascending (much smaller than reordering monthly stack)…")
    lai_y = _reorder_lon_lat_ascending(lai_y)
    etp.assert_lon_lat_strictly_increasing(lai_y, context="Annual LAI (after lon/lat reorder)")

    if args.interp_res_deg is not None:
        ddeg = float(args.interp_res_deg)
        print(f"Spatial interpolation: bilinear to ~{ddeg}° …")
        lai_y = etp.resample_lat_lon_linear_deg(lai_y, ddeg)

    etp.assert_lon_lat_strictly_increasing(
        lai_y, context="Annual LAI (before lon_edges; after interp if requested)"
    )

    lon = np.asarray(lai_y["lon"].values, dtype=np.float64)
    lat = np.asarray(lai_y["lat"].values, dtype=np.float64)
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
    transformer_ag = etp._transformer_ll_to_cea()

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
    print("Using shapefiles:\n ", p2000.name, "\n ", p2020.name, "\n ", psub.name)

    id_kw = None if args.city_id_col.strip().lower() == "auto" else args.city_id_col
    g0 = etp.add_unified_city_id(etp.prepare_geoms_wgs84(gpd.read_file(p2000)), id_kw)
    g2 = etp.add_unified_city_id(etp.prepare_geoms_wgs84(gpd.read_file(p2020)), id_kw)
    gs = etp.add_unified_city_id(etp.prepare_geoms_wgs84(gpd.read_file(psub)), id_kw)

    g0 = shift_lon_geom(g0)
    g2 = shift_lon_geom(g2)
    gs = shift_lon_geom(gs)

    print("Building Core / Expansion / Peri …")
    city_geoms = etp.build_core_expansion_peri(g0, g2, gs, etp.CITY_ID_COL)
    d2u = etp.dissolve_by_city(g2, etp.CITY_ID_COL)
    d2u = d2u.rename(columns={"geometry": "geometry_2020_full"})
    city_geoms = city_geoms.merge(
        d2u[[etp.CITY_ID_COL, "geometry_2020_full"]], on=etp.CITY_ID_COL, how="left"
    )

    mask_fb: float | None = (
        None
        if args.mask_fallback_threshold < 0
        else float(args.mask_fallback_threshold)
    )
    if mask_fb is not None:
        _fb_desc = "any overlap" if mask_fb <= 0 else str(mask_fb)
        print(
            f"Mask fallback on: primary area fraction={args.area_threshold}, "
            f"fallback when no pixels: {_fb_desc}"
        )
    else:
        print("Mask fallback off (--mask-fallback-threshold -1); expect more NaNs.")

    panel_path = args.out_dir / "panel_city_year_region_LAI.csv"
    overall_path = args.out_dir / "city_year_overall_LAI.csv"
    fpanel = foverall = True

    by_dir = args.out_dir / "by_city"
    by_dir.mkdir(parents=True, exist_ok=True)

    print(
        "Zonal and city-wide: per-city write (master CSV appended + by_city/), "
        "same pattern as aggregate_etp_to_city_scale.py …"
    )
    n_fail = 0
    for _, geom_row in tqdm(
        city_geoms.iterrows(), total=len(city_geoms), desc="cities"
    ):
        cid = geom_row[etp.CITY_ID_COL]
        safe = etp.safe_city_filename(cid)
        try:
            rec_p = records_lai_region_panel(
                geom_row,
                etp.CITY_ID_COL,
                lai_y,
                lon_edges,
                lat_edges,
                weight_grid,
                args.area_threshold,
                mask_fb,
                transformer_ag,
                years,
            )
            rec_o = records_lai_city_overall(
                geom_row,
                etp.CITY_ID_COL,
                lai_y,
                lon_edges,
                lat_edges,
                weight_grid,
                args.area_threshold,
                mask_fb,
                transformer_ag,
                years,
            )
            if args.impute_core_lai_from_overall_when_core_empty:
                rec_p = impute_core_lai_from_overall_if_core_geom_empty(
                    rec_p, rec_o, geom_row
                )
        except Exception as exc:  # noqa: BLE001 — per-city failures still yield placeholder outputs
            n_fail += 1
            err_txt = f"{type(exc).__name__}: {exc}\n\n{traceback.format_exc()}"
            (by_dir / f"{safe}_FAILED.txt").write_text(err_txt, encoding="utf-8")
            rec_p = _nan_lai_panel_records(cid, years, etp.CITY_ID_COL)
            rec_o = _nan_lai_overall_records(cid, years, etp.CITY_ID_COL)

        df_p = pd.DataFrame.from_records(rec_p)
        etp._append_df_csv(panel_path, df_p, first=fpanel)
        fpanel = False
        df_o = pd.DataFrame.from_records(rec_o)
        etp._append_df_csv(overall_path, df_o, first=foverall)
        foverall = False

        df_p.to_csv(
            by_dir / f"{safe}_lai_panel_city_year_region.csv", index=False
        )
        df_o.to_csv(
            by_dir / f"{safe}_lai_city_year_overall.csv", index=False
        )

    n_city = len(city_geoms)
    print(
        f"Per-city CSV done: {n_city} cities; "
        f"OK {n_city - n_fail}, failed with NaN placeholder (see *_FAILED.txt): {n_fail}."
    )
    print("Written (append-by-city master):", panel_path)
    print("Written (append-by-city master):", overall_path)

    df_panel = pd.read_csv(panel_path)
    df_overall = pd.read_csv(overall_path)

    clim = etp.panel_to_region_climatology(df_panel, ["LAI"])

    y0, y1 = int(args.trend_year_start), int(args.trend_year_end)
    if y1 < y0:
        y0, y1 = y1, y0
    print(f"Computing {y0}–{y1} LAI trends (Theil–Sen, LAI/year) and window means …")
    trend_overall = lai_trends_by_group(
        df_overall,
        ["city_id"],
        year_start=y0,
        year_end=y1,
    )
    trend_panel = lai_trends_by_group(
        df_panel,
        ["city_id", "region"],
        year_start=y0,
        year_end=y1,
    )

    trend_overall_path = args.out_dir / "city_lai_trend_overall.csv"
    trend_panel_path = args.out_dir / "city_lai_trend_by_region.csv"
    trend_overall.to_csv(trend_overall_path, index=False)
    trend_panel.to_csv(trend_panel_path, index=False)

    sheet_trend_o = f"trend_overall_{y0}_{y1}"[:31]
    sheet_trend_p = f"trend_region_{y0}_{y1}"[:31]

    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as w:
        df_panel.to_excel(w, sheet_name="panel_city_year_region", index=False)
        df_overall.to_excel(w, sheet_name="city_year_overall", index=False)
        clim.to_excel(w, sheet_name="region_climatology_mean", index=False)
        trend_overall.to_excel(w, sheet_name=sheet_trend_o, index=False)
        trend_panel.to_excel(w, sheet_name=sheet_trend_p, index=False)

    meta = args.out_dir / "RUN_METADATA_LAI.txt"
    meta.write_text(
        "Monthly LAI → annual mean (calendar year) → lon 0..360 / lat ascending aligned with polygons → "
        "cos(lat)-weighted means over core/expansion/peri and overall union.\n"
        "Excel: yearly panel/overall sheets; climatologies and Theil–Sen LAI_mean / "
        "LAI_trend_per_year on [trend_year_start, trend_year_end].\n"
        f"lai_dir: {args.lai_dir}\n"
        f"var: {args.var_lai}\n"
        f"shp_dir: {args.shp_dir}\n"
        f"interp_res_deg: {args.interp_res_deg}\n"
        f"impute_core_lai_from_overall_when_core_empty: {args.impute_core_lai_from_overall_when_core_empty}\n"
        f"area_threshold: {args.area_threshold}\n"
        f"mask_fallback_threshold: {args.mask_fallback_threshold}\n"
        f"trend_years: {y0}–{y1}\n"
        f"panel_city_year_region_LAI (append): {panel_path}\n"
        f"city_year_overall_LAI (append): {overall_path}\n"
        f"excel: {xlsx_path}\n"
        f"csv_trend_overall: {trend_overall_path}\n"
        f"csv_trend_region: {trend_panel_path}\n"
        f"n_cities: {n_city}; n_city_failures_nan_placeholder: {n_fail}\n"
        "(failed cities emit NaN CSVs and *_FAILED.txt with traceback)\n",
        encoding="utf-8",
    )
    print("Written Excel:", xlsx_path)
    print("City-wide appended CSV (same convention as ET/P script):", panel_path, overall_path)
    print("Per-city yearly CSV under:", by_dir)
    print("City-wide trends CSV:", trend_overall_path)
    print("Zonal trends CSV:", trend_panel_path)
    print("Metadata:", meta)


if __name__ == "__main__":
    main()
