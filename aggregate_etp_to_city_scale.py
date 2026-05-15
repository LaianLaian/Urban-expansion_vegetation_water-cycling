#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Aggregate global grids W1=ET/P, W2=ET_vegenon/P to urban water-cycle diagnostics.

Optionally ingest annual ET, ET_vegenon and precipitation P, aggregate under the **same masks**,
and emit separate CSVs to attribute ratio changes to ET vs P pathways.

Methods:
- ``V = W1 - W2`` contrasts changing vegetation versus vegetation fixed around 2000 on ET/P; this is scenario
  modulation, **not** a strict causal decomposition.
- Urban signatures are characterised via **core / expansion / peri** structure and diagnostics such as **Expansion**,
  not standalone experimental differencing.

Dependencies: numpy, pandas, xarray, geopandas, shapely, rasterio, pyproj, scipy
Optional: tqdm

Examples:
  python aggregate_etp_to_city_scale.py --out-dir ../city_etp_outputs
  python aggregate_etp_to_city_scale.py --interp-res-deg 0.05 --area-threshold 0.25
  python aggregate_etp_to_city_scale.py --mask-fallback-threshold -1
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
from pyproj import CRS, Transformer
from scipy.stats import theilslopes
from shapely.geometry import Polygon, box, mapping
from shapely.geometry.base import BaseGeometry
from shapely.ops import transform as shp_transform, unary_union
from shapely.validation import make_valid

try:
    from shapely.errors import GEOSException
except ImportError:
    GEOSException = Exception

try:
    from tqdm import tqdm
except ImportError:

    def tqdm(x, **kwargs):
        return x


PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CEA_PROJ4 = "+proj=cea +lat_ts=0 +lon_0=0 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"
# Unified joining key copied from whichever source attribute exists across shapefiles
CITY_ID_COL = "city_id"

# Candidate columns when guessing the city unique ID (e.g. origfid vs ORIG_FID)
CITY_ID_SOURCE_CANDIDATES = (
    "ORIG_FID",
    "origfid",
    "Orig_Fid",
    "ORIGFID",
    "OrigFID",
    "ct2000_id",
    "city2000_id",
    "ORIG_FID_1",
    "FID",
    "OBJECTID",
)


def resolve_city_id_source_column(columns: list[str], explicit: str | None) -> str:
    """Column name carrying the municipality unique identifier in each shapefile."""
    non_geom = [c for c in columns if str(c).lower() != "geometry"]
    if explicit and explicit.strip().lower() != "auto":
        if explicit not in columns:
            raise KeyError(
                f"City ID column '{explicit}' not found among columns {list(columns)}"
            )
        return explicit
    lower_map = {str(c).lower(): c for c in non_geom}
    for cand in CITY_ID_SOURCE_CANDIDATES:
        if cand in columns:
            return cand
        lc = cand.lower()
        if lc in lower_map:
            return lower_map[lc]
    raise ValueError(
        "Could not infer city ID column automatically; pass --city-id-col "
        "(e.g. origfid or ORIG_FID). "
        f"Available columns: {list(columns)}"
    )


def add_unified_city_id(gdf: gpd.GeoDataFrame, explicit: str | None) -> gpd.GeoDataFrame:
    """Copy inferred source column onto ``city_id`` (does not drop the original attribute)."""
    col = resolve_city_id_source_column(list(gdf.columns), explicit)
    out = gdf.copy()
    out[CITY_ID_COL] = out[col]
    return out


def _standardize_lon(lon: np.ndarray, lon_axis: str = "0..360") -> np.ndarray:
    x = np.asarray(lon, dtype=np.float64)
    if lon_axis == "0..360":
        return (x + 360.0) % 360.0
    return x


def load_w1_w2_v(
    path_w1: Path,
    path_w2: Path,
    var_w1: str,
    var_w2: str,
) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray]:
    ds1 = xr.open_dataset(path_w1)
    ds2 = xr.open_dataset(path_w2)
    w1 = ds1[var_w1]
    w2 = ds2[var_w2]
    if w1.dims != w2.dims or w1.shape != w2.shape:
        raise ValueError("W1 and W2 dimensional layout or shapes differ.")
    # Align coordinates
    w2 = w2.reindex_like(w1, method=None)
    w1, w2 = xr.align(w1, w2, join="inner")
    v = w1 - w2
    v.name = "V"
    # Enforce ordering time, lat, lon
    order = ("time", "lat", "lon")
    for d in order:
        if d not in w1.dims:
            raise ValueError(f"Missing dimension {d}; current dims = {w1.dims}")
    w1 = w1.transpose(*order)
    w2 = w2.transpose(*order)
    v = v.transpose(*order)
    ds1.close()
    ds2.close()
    return w1, w2, v


def _normalize_to_time_lat_lon(da: xr.DataArray) -> xr.DataArray:
    d = set(da.dims)
    if d == {"time", "lat", "lon"}:
        return da.transpose("time", "lat", "lon")
    if d == {"lon", "lat", "time"}:
        return da.transpose("time", "lat", "lon")
    if d == {"lat", "lon", "time"}:
        return da.transpose("time", "lat", "lon")
    raise ValueError(f"Cannot reshape to time,lat,lon — dims={da.dims}")


def load_annual_field_aligned(
    path: Path | None,
    var: str,
    ref: xr.DataArray,
    decode_times: bool | None = None,
) -> xr.DataArray | None:
    """Load an annual raster and snap it onto ``ref`` (time/lat/lon); return None when the file path is absent."""
    if path is None or not path.exists():
        return None
    kw = {}
    if decode_times is not None:
        kw["decode_times"] = decode_times
    ds = xr.open_dataset(path, **kw)
    if var not in ds:
        ds.close()
        raise KeyError(f"{path}: variable {var} missing; available={list(ds.data_vars)}")
    da = _normalize_to_time_lat_lon(ds[var])
    ds.close()
    return align_annual_field_to_w1_grid(da, ref, context=str(path))


def align_annual_field_to_w1_grid(
    da: xr.DataArray,
    ref: xr.DataArray,
    *,
    context: str = "grid",
) -> xr.DataArray:
    """
    Align yearly ET / ET_vegenon / precipitation (``da``) to W1 coordinates (``ref`` time/lat/lon) consistent with polygon masks derived from lon_edges.

    - If latitude is descending (e.g., 90° toward South pole), ``sortby('lat')`` first.
    - If ``da`` spans -180..180 (such as MSWEP from -179.95) but ``ref`` is 0..360, remap longitude then ``sortby('lon')``.
    - Conversely, if ``ref`` is -180..180 while ``da`` is 0..360, unwrap and sort (rare).
    - Finally ``interp`` if 1-D coordinates still mismatch after alignment.
    """
    da = _normalize_to_time_lat_lon(da)
    ref = _normalize_to_time_lat_lon(ref)

    if da.sizes.get("time") != ref.sizes.get("time"):
        raise ValueError(
            f"{context}: time dimension length {da.sizes.get('time')} "
            f"does not equal W1 length {ref.sizes.get('time')}"
        )

    lat = np.asarray(da["lat"].values, dtype=np.float64)
    if lat.size >= 2 and lat[0] > lat[-1]:
        da = da.sortby("lat")

    lon = np.asarray(da["lon"].values, dtype=np.float64)
    lon_ref = np.asarray(ref["lon"].values, dtype=np.float64)
    da_ax = infer_lon_axis(lon)
    ref_ax = infer_lon_axis(lon_ref)

    if da_ax == "-180..180" and ref_ax == "0..360":
        da = da.assign_coords(lon=("lon", lon_to_0_360(lon)))
        da = da.sortby("lon")
    elif da_ax == "0..360" and ref_ax == "-180..180":
        lon_new = np.where(lon > 180.0, lon - 360.0, lon)
        da = da.assign_coords(lon=("lon", lon_new))
        da = da.sortby("lon")

    lat_da = np.asarray(da["lat"].values, dtype=np.float64)
    lon_da = np.asarray(da["lon"].values, dtype=np.float64)
    lat_r = np.asarray(ref["lat"].values, dtype=np.float64)
    lon_r = np.asarray(ref["lon"].values, dtype=np.float64)

    lat_match = lat_da.shape == lat_r.shape and np.allclose(
        lat_da, lat_r, rtol=0.0, atol=1e-3
    )
    lon_match = lon_da.shape == lon_r.shape and np.allclose(
        lon_da, lon_r, rtol=0.0, atol=1e-3
    )

    if lat_match and lon_match:
        da = da.assign_coords(lat=ref["lat"], lon=ref["lon"])
    else:
        da = da.interp(lat=ref["lat"], lon=ref["lon"], method="linear")

    da = da.assign_coords(time=ref["time"])
    return _normalize_to_time_lat_lon(da)


def resample_lat_lon_linear_deg(da: xr.DataArray, ddeg: float) -> xr.DataArray:
    """
    Bilinear interpolation of ``time``/Lat/Lon grids onto regular lon/lat spacing ``ddeg`` (degrees).

    Helps refine coarse grids before masking to reduce NaNs from coarse cells + fractional-area thresholds.
    Extent stays within the bounding box of source data (~0.05° global ⇒ ~7200 × 3600 cells); watch RAM.
    """
    if ddeg <= 0:
        raise ValueError("--interp-res-deg must be positive")
    da = _normalize_to_time_lat_lon(da)
    lat = np.asarray(da["lat"].values, dtype=np.float64)
    lon = np.asarray(da["lon"].values, dtype=np.float64)
    lat_min, lat_max = float(np.nanmin(lat)), float(np.nanmax(lat))
    lon_min, lon_max = float(np.nanmin(lon)), float(np.nanmax(lon))
    if lat_max < lat_min:
        lat_min, lat_max = lat_max, lat_min
    nlat = max(2, int(round((lat_max - lat_min) / ddeg)) + 1)
    nlon = max(2, int(round((lon_max - lon_min) / ddeg)) + 1)
    new_lat = np.linspace(lat_min, lat_max, nlat, dtype=np.float64)
    new_lon = np.linspace(lon_min, lon_max, nlon, dtype=np.float64)
    out = da.interp(lat=new_lat, lon=new_lon, method="linear")
    return _normalize_to_time_lat_lon(out)


def try_load_annual_first_match(
    path: Path,
    var_candidates: list[str],
    ref: xr.DataArray,
) -> xr.DataArray | None:
    """Try each candidate variable × decode_times flag; first success returns aligned data."""
    if not path.exists():
        return None
    seen: set[str] = set()
    ordered = []
    for v in var_candidates:
        if v not in seen:
            seen.add(v)
            ordered.append(v)
    for dec in (True, False):
        for var in ordered:
            try:
                da = load_annual_field_aligned(path, var, ref, decode_times=dec)
                if da is not None:
                    return da
            except (KeyError, ValueError, OSError, Exception):
                continue
    return None


def infer_lon_axis(lon: np.ndarray) -> str:
    if np.nanmax(lon) > 180.5:
        return "0..360"
    return "-180..180"


def lon_to_0_360(lon_1d: np.ndarray) -> np.ndarray:
    return (np.asarray(lon_1d, dtype=np.float64) + 360.0) % 360.0


def assert_lon_lat_strictly_increasing(
    da: xr.DataArray, *, context: str = "grid"
) -> None:
    """
    Validate strictly increasing latitude/longitude axes (positive adjacent differences).
    Raster masks / cosine weights assume monotonic edges; duplicated or unordered coordinates misalign overlays.
    """
    da = _normalize_to_time_lat_lon(da)
    lat = np.asarray(da["lat"].values, dtype=np.float64)
    lon = np.asarray(da["lon"].values, dtype=np.float64)
    if lat.size >= 2:
        dlat = np.diff(lat)
        if not np.all(dlat > 0):
            i = int(np.flatnonzero(dlat <= 0)[0])
            raise ValueError(
                f"{context}: latitude must strictly increase — between index {i} and {i + 1} "
                f"lat={float(lat[i]):.8g} → {float(lat[i + 1]):.8g}."
            )
    if lon.size >= 2:
        dlon = np.diff(lon)
        if not np.all(dlon > 0):
            i = int(np.flatnonzero(dlon <= 0)[0])
            raise ValueError(
                f"{context}: longitude must strictly increase — between index {i} and {i + 1} "
                f"lon={float(lon[i]):.8g} → {float(lon[i + 1]):.8g}."
            )


def prepare_geoms_wgs84(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if gdf.crs is None:
        gdf = gdf.set_crs(4326)
    else:
        gdf = gdf.to_crs(4326)
    # repair invalid geometries
    gdf = gdf.copy()
    gdf["geometry"] = gdf.geometry.apply(lambda g: make_valid(g) if g is not None else g)
    return gdf


def clean_geom(geom: BaseGeometry | None) -> BaseGeometry:
    """Heal invalid/topology errors; coerce GeometryCollections toward polygonal parts."""
    if geom is None or geom.is_empty:
        return Polygon()
    try:
        g = make_valid(geom)
    except Exception:
        g = geom
    try:
        if not g.is_valid:
            g = g.buffer(0)
    except Exception:
        pass
    if g.geom_type == "GeometryCollection":
        polys = [x for x in g.geoms if x.geom_type in ("Polygon", "MultiPolygon")]
        g = unary_union(polys) if polys else Polygon()
    return g


def safe_intersection(a: BaseGeometry, b: BaseGeometry, *, context: str = "") -> BaseGeometry:
    a, b = clean_geom(a), clean_geom(b)
    if a.is_empty or b.is_empty:
        return Polygon()
    try:
        import shapely

        if hasattr(shapely, "intersection"):
            return shapely.intersection(a, b, grid_size=1e-6)
    except Exception:
        pass
    try:
        return a.intersection(b)
    except (GEOSException, Exception):
        try:
            return a.buffer(0).intersection(b.buffer(0))
        except (GEOSException, Exception):
            warnings.warn(f"Intersection failed {context}; returning empty Polygon.")
            return Polygon()


def safe_difference(a: BaseGeometry, b: BaseGeometry, *, context: str = "") -> BaseGeometry:
    """
    Robust geometric difference alleviating GEOS TopologyException cases (bad rings, seams near lon 0°/360°).
    Falls back to returning ``a`` unchanged if every attempt fails — keeps the pipeline alive at the cost of perfect exclusivity.
    """
    a, b = clean_geom(a), clean_geom(b)
    if a.is_empty:
        return Polygon()
    if b.is_empty:
        return a
    try:
        import shapely

        if hasattr(shapely, "difference"):
            out = shapely.difference(a, b, grid_size=1e-6)
            if out is not None and not out.is_empty:
                return out
    except (GEOSException, Exception):
        pass
    try:
        return a.difference(b)
    except (GEOSException, Exception):
        pass
    try:
        return a.difference(b.simplify(1e-5, preserve_topology=True))
    except (GEOSException, Exception):
        pass
    try:
        return a.difference(b.buffer(1e-7))
    except (GEOSException, Exception):
        pass
    try:
        return a.difference(b.buffer(-1e-7))
    except (GEOSException, Exception):
        pass
    warnings.warn(
        f"Difference failed {context}; keeping minuend geometry (check polygons crossing the dateline / topology)."
    )
    return a


def dissolve_by_city(gdf: gpd.GeoDataFrame, id_col: str) -> gpd.GeoDataFrame:
    if id_col not in gdf.columns:
        raise KeyError(f"No column `{id_col}` in GeoDataFrame; columns={list(gdf.columns)}")
    return gdf.dissolve(by=id_col, as_index=False)


def build_core_expansion_peri(
    gdf_2000: gpd.GeoDataFrame,
    gdf_2020: gpd.GeoDataFrame,
    gdf_sub: gpd.GeoDataFrame,
    id_col: str,
) -> gpd.GeoDataFrame:
    """
    One row per city_id with geometry_core, geometry_expansion, geometry_peri.
    Core = U2000 ∩ U2020; Expansion = U2020 − U2000; Peri = suburban ring with overlaps removed against U2020 and expansion so the three buckets stay mutually exclusive.
    """
    d0 = dissolve_by_city(gdf_2000, id_col).set_index(id_col)
    d2 = dissolve_by_city(gdf_2020, id_col).set_index(id_col)
    ds = dissolve_by_city(gdf_sub, id_col).set_index(id_col)

    all_ids = sorted(set(d0.index) | set(d2.index) | set(ds.index))
    rows = []
    for cid in tqdm(all_ids, desc="core/expansion/peri"):
        g0 = d0.loc[cid].geometry if cid in d0.index else None
        g2 = d2.loc[cid].geometry if cid in d2.index else None
        gsb = ds.loc[cid].geometry if cid in ds.index else None

        g0 = g0 if g0 is not None and not g0.is_empty else Polygon()
        g2 = g2 if g2 is not None and not g2.is_empty else Polygon()

        ctx_base = f"city_id={cid}"

        if not g0.is_empty and not g2.is_empty:
            core_i = safe_intersection(g0, g2, context=f"{ctx_base} core")
        else:
            core_i = Polygon()

        if not g2.is_empty:
            if not g0.is_empty:
                exp = safe_difference(g2, g0, context=f"{ctx_base} expansion")
            else:
                exp = clean_geom(g2)
        else:
            exp = Polygon()

        peri = gsb if gsb is not None and not gsb.is_empty else Polygon()
        peri = clean_geom(peri)
        # make peri/expansion pairwise exclusive with safer differences
        if not peri.is_empty and not g2.is_empty:
            peri = safe_difference(peri, g2, context=f"{ctx_base} peri\\U2020")
        if not peri.is_empty and not exp.is_empty:
            peri = safe_difference(peri, exp, context=f"{ctx_base} peri\\exp")
        if not exp.is_empty and not peri.is_empty:
            exp = safe_difference(exp, peri, context=f"{ctx_base} exp\\peri")

        rows.append(
            {
                id_col: cid,
                "geometry_core": core_i if not core_i.is_empty else Polygon(),
                "geometry_expansion": exp if not exp.is_empty else Polygon(),
                "geometry_peri": peri if not peri.is_empty else Polygon(),
            }
        )
    return gpd.GeoDataFrame(rows, geometry="geometry_core", crs=4326)


def _transformer_ll_to_cea() -> Transformer:
    return Transformer.from_crs(CRS.from_epsg(4326), CRS.from_proj4(CEA_PROJ4), always_xy=True)


def ll_poly_to_cea(geom, transformer: Transformer):
    if geom is None or geom.is_empty:
        return geom
    return shp_transform(transformer.transform, geom)


def grid_cell_box(lon_edges: np.ndarray, lat_edges: np.ndarray, i: int, j: int) -> Polygon:
    return box(
        float(lon_edges[j]),
        float(lat_edges[i]),
        float(lon_edges[j + 1]),
        float(lat_edges[i + 1]),
    )


def _intersection_area_fraction_in_cea(
    poly_cea: BaseGeometry,
    cell_cea: BaseGeometry,
    poly_ll: BaseGeometry,
    cell_ll: BaseGeometry,
) -> float:
    """
    Fraction of raster cell overlapped by the polygon measured in cylindrical equal-area (CEA) space.
    On GEOS failures, fall back to WGS84 heuristics and centroid containment checks.
    """
    poly_cea = clean_geom(poly_cea)
    cell_cea = clean_geom(cell_cea)
    a_cell = float(cell_cea.area)
    if a_cell <= 0:
        return 0.0

    def _try_cea_inter() -> float | None:
        try:
            import shapely

            inter = shapely.intersection(poly_cea, cell_cea, grid_size=1e-3)
            return 0.0 if inter.is_empty else float(inter.area)
        except (GEOSException, Exception):
            pass
        try:
            inter = poly_cea.intersection(cell_cea)
            return 0.0 if inter.is_empty else float(inter.area)
        except (GEOSException, Exception):
            pass
        try:
            inter = poly_cea.buffer(0).intersection(cell_cea.buffer(0))
            return 0.0 if inter.is_empty else float(inter.area)
        except (GEOSException, Exception):
            return None

    ia = _try_cea_inter()
    if ia is not None:
        return min(1.0, max(0.0, ia / a_cell))

    # WGS84 area ratio (non-equal-area approximation)
    try:
        inter_ll = safe_intersection(poly_ll, cell_ll, context="raster_frac_ll")
        a_ll = float(cell_ll.area)
        if a_ll > 0 and not inter_ll.is_empty:
            return min(1.0, max(0.0, float(inter_ll.area) / a_ll))
    except (GEOSException, Exception):
        pass

    try:
        c = cell_ll.centroid
        pl = clean_geom(poly_ll)
        if pl.covers(c) or pl.contains(c):
            return 1.0
    except (GEOSException, Exception):
        pass
    return 0.0


def _fraction_passes_threshold(frac: float, threshold: float) -> bool:
    """For threshold>0 require overlap fraction ≥ threshold; for threshold≤0 trigger on any overlap (helps tiny patches)."""
    if threshold <= 0:
        return frac > 1e-14
    return frac >= threshold


def rasterize_fraction_mask(
    poly_ll: Polygon,
    lon_edges: np.ndarray,
    lat_edges: np.ndarray,
    transformer: Transformer,
    threshold: float = 0.5,
    poly_cea_cache: Polygon | None = None,
) -> tuple[np.ndarray, slice, slice]:
    """
    On cells intersecting polygon bounds compute overlap fraction vs polygon (CEA true area ratio).
    Cells exceeding ``threshold`` become True — see `_fraction_passes_threshold`. Returns tuple (mask, row_slice, col_slice).
    To save RAM only the clipped window mask is returned alongside slices referencing the global grid index.
    """
    if poly_ll.is_empty:
        return np.zeros((0, 0), dtype=bool), slice(0, 0), slice(0, 0)

    minx, miny, maxx, maxy = poly_ll.bounds
    # search lon edges spanning 0..360
    j0 = int(np.searchsorted(lon_edges, minx, side="right") - 1)
    j1 = int(np.searchsorted(lon_edges, maxx, side="left") + 1)
    i0 = int(np.searchsorted(lat_edges, miny, side="right") - 1)
    i1 = int(np.searchsorted(lat_edges, maxy, side="left") + 1)
    j0 = max(0, j0)
    j1 = min(len(lon_edges) - 1, j1)
    i0 = max(0, i0)
    i1 = min(len(lat_edges) - 1, i1)
    if j1 <= j0 or i1 <= i0:
        return np.zeros((0, 0), dtype=bool), slice(0, 0), slice(0, 0)

    poly_cea = poly_cea_cache if poly_cea_cache is not None else ll_poly_to_cea(poly_ll, transformer)
    poly_cea = clean_geom(poly_cea)
    if poly_cea.is_empty:
        return np.zeros((i1 - i0, j1 - j0), dtype=bool), slice(i0, i1), slice(j0, j1)

    poly_ll = clean_geom(poly_ll)
    out = np.zeros((i1 - i0, j1 - j0), dtype=bool)
    for ii in range(i0, i1):
        for jj in range(j0, j1):
            cell_ll = grid_cell_box(lon_edges, lat_edges, ii, jj)
            cell_ll = clean_geom(cell_ll)
            cell_cea = ll_poly_to_cea(cell_ll, transformer)
            frac = _intersection_area_fraction_in_cea(poly_cea, cell_cea, poly_ll, cell_ll)
            if _fraction_passes_threshold(frac, threshold):
                out[ii - i0, jj - j0] = True
    return out, slice(i0, i1), slice(j0, j1)


def rasterize_fraction_mask_with_fallback(
    poly_ll: Polygon,
    lon_edges: np.ndarray,
    lat_edges: np.ndarray,
    transformer: Transformer,
    threshold: float,
    fallback_threshold: float | None,
) -> tuple[np.ndarray, slice, slice]:
    """
    Apply ``threshold`` fraction mask first; if polygon is nonempty but yields no True pixels, optionally retry ``fallback_threshold``.
    Skip retries when fallback is ``None``, negative, or numerically identical to ``threshold``.
    """
    mask_blk, sl_r, sl_c = rasterize_fraction_mask(
        poly_ll, lon_edges, lat_edges, transformer, threshold=threshold
    )
    if poly_ll.is_empty:
        return mask_blk, sl_r, sl_c
    if mask_blk.size > 0 and np.any(mask_blk):
        return mask_blk, sl_r, sl_c
    if fallback_threshold is None or fallback_threshold < 0:
        return mask_blk, sl_r, sl_c
    if abs(float(fallback_threshold) - float(threshold)) < 1e-12:
        return mask_blk, sl_r, sl_c
    return rasterize_fraction_mask(
        poly_ll,
        lon_edges,
        lat_edges,
        transformer,
        threshold=float(fallback_threshold),
    )


def cell_area_weights(lat_centers: np.ndarray, lon_centers: np.ndarray) -> np.ndarray:
    """Relative pixel weight ∝ cos(lat) representing spherical elemental area scaling for area-weighted means."""
    lat_rad = np.deg2rad(lat_centers)
    w_lat = np.cos(lat_rad)
    w = w_lat[:, np.newaxis] * np.ones((1, lon_centers.size), dtype=np.float64)
    return w


def weighted_mean_2d(
    values: np.ndarray,
    weights: np.ndarray,
    mask: np.ndarray,
) -> float:
    """``values``, ``weights`` and ``mask`` share shape; ignores ``values`` NaNs."""
    m = mask & np.isfinite(values)
    if not np.any(m):
        return np.nan
    w = weights[m]
    x = values[m]
    sw = np.nansum(w)
    if sw <= 0:
        return np.nan
    return float(np.nansum(x * w) / sw)


def safe_city_filename(cid: object) -> str:
    """Filesystem-safe basename for exporting per-city artefacts."""
    s = str(cid).strip()
    for bad in ("\\", "/", ":", "*", "?", '"', "<", ">", "|"):
        s = s.replace(bad, "_")
    return s or "unknown_id"


def _panel_years_from_w1(w1: xr.DataArray) -> np.ndarray:
    years = np.asarray(w1["time"].values)
    if years.dtype == object or np.issubdtype(years.dtype, np.datetime64):
        years = pd.to_datetime(years).year.values
    return years


def records_for_city_region_panel(
    row: pd.Series,
    id_col: str,
    w1: xr.DataArray,
    w2: xr.DataArray,
    v: xr.DataArray,
    lon_edges: np.ndarray,
    lat_edges: np.ndarray,
    weight_grid: np.ndarray,
    threshold: float,
    mask_fallback_threshold: float | None,
    extras: dict[str, xr.DataArray],
    transformer: Transformer,
    years: np.ndarray,
) -> list[dict]:
    """Per city: yearly records covering core/expansion/peri."""
    cid = row[id_col]
    regions = ("core", "expansion", "peri")
    geom_cols = ("geometry_core", "geometry_expansion", "geometry_peri")
    records: list[dict] = []
    for reg, gcol in zip(regions, geom_cols):
        poly = row[gcol]
        poly = clean_geom(poly)
        mask_blk, sl_r, sl_c = rasterize_fraction_mask_with_fallback(
            poly,
            lon_edges,
            lat_edges,
            transformer,
            threshold,
            mask_fallback_threshold,
        )
        if mask_blk.size == 0 or not np.any(mask_blk):
            for yi, yv in enumerate(years):
                rec = {
                    "city_id": cid,
                    "year": int(yv),
                    "region": reg,
                    "W1": np.nan,
                    "W2": np.nan,
                    "V": np.nan,
                }
                for k in extras:
                    rec[k] = np.nan
                records.append(rec)
            continue

        w_sub = weight_grid[sl_r, sl_c]
        for yi, yv in enumerate(years):
            a1 = w1.isel(time=yi).values[sl_r, sl_c]
            a2 = w2.isel(time=yi).values[sl_r, sl_c]
            av = v.isel(time=yi).values[sl_r, sl_c]
            rec = {
                "city_id": cid,
                "year": int(yv),
                "region": reg,
                "W1": weighted_mean_2d(a1, w_sub, mask_blk),
                "W2": weighted_mean_2d(a2, w_sub, mask_blk),
                "V": weighted_mean_2d(av, w_sub, mask_blk),
            }
            for k, da in extras.items():
                ax = da.isel(time=yi).values[sl_r, sl_c]
                rec[k] = weighted_mean_2d(ax, w_sub, mask_blk)
            records.append(rec)
    return records


def records_for_city_overall_union(
    row: pd.Series,
    id_col: str,
    w1: xr.DataArray,
    w2: xr.DataArray,
    v: xr.DataArray,
    lon_edges: np.ndarray,
    lat_edges: np.ndarray,
    weight_grid: np.ndarray,
    threshold: float,
    mask_fallback_threshold: float | None,
    extras: dict[str, xr.DataArray],
    transformer: Transformer,
    years: np.ndarray,
) -> list[dict]:
    """Per city-year using union-mask across the three subregions."""
    cid = row[id_col]
    parts = []
    for gcol in ("geometry_core", "geometry_expansion", "geometry_peri"):
        p = row[gcol]
        if p is None or getattr(p, "is_empty", True):
            continue
        p = clean_geom(p)
        if p.is_empty:
            continue
        parts.append(p)
    records: list[dict] = []
    if not parts:
        for yv in years:
            rec = {"city_id": cid, "year": int(yv), "W1": np.nan, "W2": np.nan, "V": np.nan}
            for k in extras:
                rec[k] = np.nan
            records.append(rec)
        return records
    poly_u = unary_union(parts)
    mask_blk, sl_r, sl_c = rasterize_fraction_mask_with_fallback(
        poly_u,
        lon_edges,
        lat_edges,
        transformer,
        threshold,
        mask_fallback_threshold,
    )
    w_sub = weight_grid[sl_r, sl_c]
    if mask_blk.size == 0 or not np.any(mask_blk):
        for yv in years:
            rec = {"city_id": cid, "year": int(yv), "W1": np.nan, "W2": np.nan, "V": np.nan}
            for k in extras:
                rec[k] = np.nan
            records.append(rec)
        return records
    for yi, yv in enumerate(years):
        a1 = w1.isel(time=yi).values[sl_r, sl_c]
        a2 = w2.isel(time=yi).values[sl_r, sl_c]
        av = v.isel(time=yi).values[sl_r, sl_c]
        rec = {
            "city_id": cid,
            "year": int(yv),
            "W1": weighted_mean_2d(a1, w_sub, mask_blk),
            "W2": weighted_mean_2d(a2, w_sub, mask_blk),
            "V": weighted_mean_2d(av, w_sub, mask_blk),
        }
        for k, da in extras.items():
            ax = da.isel(time=yi).values[sl_r, sl_c]
            rec[k] = weighted_mean_2d(ax, w_sub, mask_blk)
        records.append(rec)
    return records


def _append_df_csv(path: Path, df: pd.DataFrame, *, first: bool) -> None:
    df.to_csv(path, mode="w" if first else "a", header=first, index=False)


def panel_to_region_climatology(panel: pd.DataFrame, value_cols: list[str] | None = None) -> pd.DataFrame:
    if value_cols is None:
        value_cols = [c for c in panel.columns if c not in ("city_id", "year", "region")]
    g = panel.groupby(["city_id", "region"], as_index=False)[value_cols].mean()
    pivot = g.pivot(index="city_id", columns="region", values=value_cols)
    pivot.columns = [f"{a}_{b}" for a, b in pivot.columns]
    return pivot.reset_index()


def compute_city_attributes_one(
    geom_row: pd.Series,
    sub_overall: pd.DataFrame,
    id_col: str,
    year_early_end: int,
    year_late_start: int,
    transformer: Transformer,
) -> dict:
    """Build attribute dict for one city given its union-level table slice and geometries."""
    cid = geom_row[id_col]
    sub = sub_overall.sort_values("year")
    y = sub["year"].values.astype(float)
    v = sub["V"].values.astype(float)
    w1 = sub["W1"].values.astype(float)
    w2 = sub["W2"].values.astype(float)

    v_mean = float(np.nanmean(v))
    w1_mean = float(np.nanmean(w1))
    w2_mean = float(np.nanmean(w2))

    m = np.isfinite(v) & np.isfinite(y)
    if np.sum(m) >= 3:
        slope, _, _, _ = theilslopes(v[m], y[m])
        v_trend = float(slope)
    else:
        v_trend = np.nan

    early = sub[sub["year"] <= year_early_end]
    late = sub[sub["year"] >= year_late_start]
    v_early = float(np.nanmean(early["V"].values)) if len(early) else np.nan
    v_late = float(np.nanmean(late["V"].values)) if len(late) else np.nan

    c = geom_row["geometry_core"]
    e = geom_row["geometry_expansion"]
    p2020 = geom_row["geometry_2020_full"] if "geometry_2020_full" in geom_row.index else None
    if p2020 is None or pd.isna(p2020):
        p2020 = None
    elif hasattr(p2020, "is_empty") and p2020.is_empty:
        p2020 = None
    if p2020 is None:
        geoms = [g for g in (c, e) if g is not None and not g.is_empty]
        p2020 = unary_union(geoms) if geoms else Polygon()
    a_exp = ll_poly_to_cea(e, transformer).area if e is not None and not e.is_empty else 0.0
    a_2020 = (
        ll_poly_to_cea(p2020, transformer).area
        if p2020 is not None and not p2020.is_empty
        else np.nan
    )
    expansion_ratio = float(a_exp / a_2020) if a_2020 and a_2020 > 0 else np.nan

    return {
        id_col: cid,
        "V_mean": v_mean,
        "W1_mean": w1_mean,
        "W2_mean": w2_mean,
        "V_trend": v_trend,
        "Expansion": expansion_ratio,
        "V_early": v_early,
        "V_late": v_late,
    }


def compute_city_attributes(
    panel_overall: pd.DataFrame,
    city_geoms: gpd.GeoDataFrame,
    id_col: str,
    year_early_end: int,
    year_late_start: int,
) -> pd.DataFrame:
    """V_trend uses Theil–Sen slope vs year; Expansion = expansion area / total 2020 city area."""
    transformer = _transformer_ll_to_cea()
    rows = []
    for _, row in city_geoms.iterrows():
        cid = row[id_col]
        sub = panel_overall[panel_overall[id_col] == cid]
        rows.append(
            compute_city_attributes_one(
                row, sub, id_col, year_early_end, year_late_start, transformer
            )
        )
    return pd.DataFrame(rows)


def _theil_slope(years: np.ndarray, values: np.ndarray) -> float:
    m = np.isfinite(values) & np.isfinite(years)
    if np.sum(m) < 3:
        return float("nan")
    return float(theilslopes(values[m], years[m])[0])


def compute_city_attributes_et(
    panel_overall: pd.DataFrame,
    id_col: str,
    year_early_end: int,
    year_late_start: int,
) -> pd.DataFrame | None:
    """Multi-year summaries and trends for ET, ET_vegenon, and optionally V_ET = ET − ET_vegenon (subset of columns OK)."""
    has_et = "ET" in panel_overall.columns
    has_etv = "ET_vegenon" in panel_overall.columns
    if not has_et and not has_etv:
        return None
    rows = []
    for cid in panel_overall[id_col].unique():
        sub = panel_overall[panel_overall[id_col] == cid].sort_values("year")
        y = sub["year"].values.astype(float)
        early_m = sub["year"].values <= year_early_end
        late_m = sub["year"].values >= year_late_start
        row: dict = {id_col: cid}
        if has_et:
            et = sub["ET"].values.astype(float)
            row["ET_mean"] = float(np.nanmean(et))
            row["ET_trend"] = _theil_slope(y, et)
        else:
            row["ET_mean"] = float("nan")
            row["ET_trend"] = float("nan")
        if has_etv:
            etv = sub["ET_vegenon"].values.astype(float)
            row["ET_vegenon_mean"] = float(np.nanmean(etv))
            row["ET_vegenon_trend"] = _theil_slope(y, etv)
        else:
            row["ET_vegenon_mean"] = float("nan")
            row["ET_vegenon_trend"] = float("nan")
        if has_et and has_etv:
            et = sub["ET"].values.astype(float)
            etv = sub["ET_vegenon"].values.astype(float)
            v_et = et - etv
            row["V_ET_mean"] = float(np.nanmean(v_et))
            row["V_ET_trend"] = _theil_slope(y, v_et)
            row["V_ET_early"] = (
                float(np.nanmean(v_et[early_m])) if np.any(early_m) else float("nan")
            )
            row["V_ET_late"] = (
                float(np.nanmean(v_et[late_m])) if np.any(late_m) else float("nan")
            )
        else:
            row["V_ET_mean"] = float("nan")
            row["V_ET_trend"] = float("nan")
            row["V_ET_early"] = float("nan")
            row["V_ET_late"] = float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def compute_city_attributes_p(
    panel_overall: pd.DataFrame,
    id_col: str,
    year_early_end: int,
    year_late_start: int,
) -> pd.DataFrame | None:
    """Multi-year summaries and precipitation trends."""
    if "P" not in panel_overall.columns:
        return None
    rows = []
    for cid in panel_overall[id_col].unique():
        sub = panel_overall[panel_overall[id_col] == cid].sort_values("year")
        y = sub["year"].values.astype(float)
        pr = sub["P"].values.astype(float)
        early_m = sub["year"].values <= year_early_end
        late_m = sub["year"].values >= year_late_start
        rows.append(
            {
                id_col: cid,
                "P_mean": float(np.nanmean(pr)),
                "P_trend": _theil_slope(y, pr),
                "P_early": float(np.nanmean(pr[early_m])) if np.any(early_m) else float("nan"),
                "P_late": float(np.nanmean(pr[late_m])) if np.any(late_m) else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Aggregate gridded ET/P products to city/subregion scale (core / expansion / peri)."
    )
    ap.add_argument(
        "--w1",
        type=Path,
        default=PROJECT_ROOT / "ET_over_P_annual_2000_2021.nc",
        help="Path to W1 (ET/P scenario) NetCDF",
    )
    ap.add_argument(
        "--w2",
        type=Path,
        default=PROJECT_ROOT / "ET_vegenon_over_P_annual_2000_2021.nc",
        help="Path to W2 (fixed-vegetation ET/P) NetCDF",
    )
    ap.add_argument("--var-w1", type=str, default="ET_over_P")
    ap.add_argument("--var-w2", type=str, default="ET_vegenon_over_P")
    ap.add_argument(
        "--et-nc",
        type=Path,
        default=PROJECT_ROOT / "ET_yr.nc",
        help="Annual-changing ET cube (e.g. ET_yr.nc); skipped silently if absent",
    )
    ap.add_argument(
        "--et-vegenon-nc",
        type=Path,
        default=PROJECT_ROOT / "ETyr_vegenon.nc",
        help="Annual fixed-vegetation ET cube (e.g. ETyr_vegenon.nc)",
    )
    ap.add_argument(
        "--precip-nc",
        type=Path,
        default=PROJECT_ROOT / "MSWEP_annual_precip_2000_2021.nc",
        help="Annual precipitation on compatible grid (MSWEP, etc.)",
    )
    ap.add_argument("--var-et", type=str, default="ET", help="ET variable inside NetCDF")
    ap.add_argument("--var-et-vegenon", type=str, default="ET", help="ET_vegenon variable name")
    ap.add_argument(
        "--var-precip",
        type=str,
        default="P",
        help="Precipitation variable (P, annual_pricip typo alias, precipitation, …)",
    )
    ap.add_argument(
        "--skip-et-p",
        action="store_true",
        help="Skip raw ET/P component rasters — keep ratio-only outputs",
    )
    ap.add_argument(
        "--city-id-col",
        type=str,
        default="auto",
        help="Column with unique municipality ID (auto guesses ORIG_FID, origfid, ct2000_id, …)",
    )
    ap.add_argument(
        "--shp-dir",
        type=Path,
        default=PROJECT_ROOT / "filtered_outputs",
    )
    ap.add_argument(
        "--gub-2000",
        type=str,
        default="GUB_Global_2000_filtered.shp",
        help="Year-2000 urban footprint basename under --shp-dir",
    )
    ap.add_argument(
        "--gub-2020",
        type=str,
        default="GUB_Global_2020_filtered.shp",
        help="Year-2020 urban footprint basename",
    )
    ap.add_argument(
        "--suburban-2020",
        type=str,
        default="GUB_suburban_30km_2020.shp",
        help="Year-2020 peri-urban buffer ring basename",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=PROJECT_ROOT / "city_etp_outputs",
    )
    ap.add_argument(
        "--area-threshold",
        type=float,
        default=0.5,
        help=(
            "Minimum polygon/cell fractional overlap in CEA; small footprints may yield no usable cells when coarse grids are combined with large thresholds."
        ),
    )
    ap.add_argument(
        "--mask-fallback-threshold",
        type=float,
        default=0.0,
        help=(
            "If the primary fraction leaves zero pixels, retry with this looser cutoff; "
            "0 ⇒ any intersecting overlap counts. Use -1 to disable fallback old behaviour ⇒ more NaNs."
        ),
    )
    ap.add_argument(
        "--interp-res-deg",
        type=float,
        default=None,
        help=(
            "Optional bilinear resampling of loaded rasters onto a regular spacing of this many degrees "
            "(e.g. 0.05°). Increases fidelity but blows up RAM on global extents."
        ),
    )
    ap.add_argument(
        "--year-early-end",
        type=int,
        default=2010,
        help="Years ≤ this value averaged into V_early",
    )
    ap.add_argument(
        "--year-late-start",
        type=int,
        default=2010,
        help="Years ≥ this value averaged into V_late",
    )
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading grids …")
    w1, w2, v = load_w1_w2_v(args.w1, args.w2, args.var_w1, args.var_w2)
    lon = np.asarray(w1["lon"].values, dtype=np.float64)
    lon_axis = infer_lon_axis(lon)
    if lon_axis == "-180..180":
        lon = lon_to_0_360(lon)
        w1 = w1.assign_coords(lon=lon)
        w2 = w2.assign_coords(lon=lon)
        v = v.assign_coords(lon=lon)

    # Lon must be ascending in 0..360; assign_coords without resorting would break monotonic masks
    w1 = w1.sortby("lon").sortby("lat")
    w2 = w2.sortby("lon").sortby("lat")
    v = v.sortby("lon").sortby("lat")

    extra_fields: dict[str, xr.DataArray] = {}
    missing_extra_inputs: list[tuple[str, Path, str]] = []
    if not args.skip_et_p:
        print("Attempting to load annual ET, ET_vegenon, P rasters (same grid as W1)…")
        for _label, _pth in (
            ("ET", args.et_nc),
            ("ET_vegenon", args.et_vegenon_nc),
            ("P", args.precip_nc),
        ):
            if not _pth.exists():
                reason = "file missing"
                print(f"  [skip] {_label}: {_pth} ({reason})")
                missing_extra_inputs.append((_label, _pth, reason))
        try:
            da_et = load_annual_field_aligned(
                args.et_nc, args.var_et, w1, decode_times=False
            )
            if da_et is not None:
                extra_fields["ET"] = da_et
                print("  Loaded ET:", args.et_nc.name)
            else:
                reason = "empty read"
                print(f"  [skip] ET: {args.et_nc} ({reason})")
                missing_extra_inputs.append(("ET", args.et_nc, reason))
        except Exception as e:
            print(f"  [skip] ET: {args.et_nc} ({e})")
            missing_extra_inputs.append(("ET", args.et_nc, str(e)))
            warnings.warn(f"Skipping ET raster: {e}")
        try:
            da_etv = load_annual_field_aligned(
                args.et_vegenon_nc, args.var_et_vegenon, w1, decode_times=False
            )
            if da_etv is not None:
                extra_fields["ET_vegenon"] = da_etv
                print("  Loaded ET_vegenon:", args.et_vegenon_nc.name)
            else:
                reason = "empty read"
                print(f"  [skip] ET_vegenon: {args.et_vegenon_nc} ({reason})")
                missing_extra_inputs.append(("ET_vegenon", args.et_vegenon_nc, reason))
        except Exception as e:
            print(f"  [skip] ET_vegenon: {args.et_vegenon_nc} ({e})")
            missing_extra_inputs.append(("ET_vegenon", args.et_vegenon_nc, str(e)))
            warnings.warn(f"Skipping ET_vegenon raster: {e}")
        try:
            p_vars = [
                args.var_precip,
                "annual_pricip",
                "annual_precip",
                "precipitation",
                "P",
            ]
            da_p = try_load_annual_first_match(args.precip_nc, p_vars, w1)
            if da_p is not None:
                extra_fields["P"] = da_p
                print("  Loaded P:", args.precip_nc.name)
            else:
                reason = f"empty read (tried {', '.join(p_vars)})"
                print(f"  [skip] P: {args.precip_nc} ({reason})")
                missing_extra_inputs.append(("P", args.precip_nc, reason))
        except Exception as e:
            print(f"  [skip] P: {args.precip_nc} ({e})")
            missing_extra_inputs.append(("P", args.precip_nc, str(e)))
            warnings.warn(f"Skipping precipitation: {e}")
        if missing_extra_inputs:
            print("  Missing optional ET/P component inputs:")
            for _label, _pth, _reason in missing_extra_inputs:
                print(f"    - {_label}: {_pth} ({_reason})")

    if args.interp_res_deg is not None:
        ddeg = float(args.interp_res_deg)
        nlat0 = int(w1.sizes.get("lat", 0))
        nlon0 = int(w1.sizes.get("lon", 0))
        print(f"Bilinear interpolation to ~{ddeg}° regular grid (source grid ~{nlat0}×{nlon0})…")
        w1 = resample_lat_lon_linear_deg(w1, ddeg)
        w2 = resample_lat_lon_linear_deg(w2, ddeg)
        v = w1 - w2
        v.name = "V"
        for _k in list(extra_fields.keys()):
            extra_fields[_k] = resample_lat_lon_linear_deg(extra_fields[_k], ddeg)
        nlat1 = int(w1.sizes["lat"])
        nlon1 = int(w1.sizes["lon"])
        ncell = nlat1 * nlon1
        print(f"  After interpolation: grid {nlat1}×{nlon1} (~{ncell:,} px per year)")
        if ncell > 8_000_000:
            warnings.warn(
                f"Interpolation yields {ncell:,} pixels — likely heavy RAM usage; widen --interp-res-deg or crop extents."
            )

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
    # When raster lon already lives in 0..360, avoid modulo on full edge array (breaks monotonicity near 0°)
    if np.nanmin(lon) >= 0 and np.nanmax(lon) <= 360.0:
        lon_edges[0] = max(0.0, float(lon_edges[0]))
        lon_edges[-1] = min(360.0, float(lon_edges[-1]))
    else:
        lon_edges = (lon_edges + 360.0) % 360.0
        if lon_edges[-1] < lon_edges[0]:
            warnings.warn("Unusual longitude bounds (possible Greenwich crossing); verify inputs.")

    weight_grid = cell_area_weights(lat, lon)

    extra_arg = extra_fields if extra_fields else None
    print(
        "  Extra gridded fields merged into panel_city_year_region.csv / city_year_overall.csv:",
        list(extra_fields.keys()) if extra_fields else "none (only W1, W2, V)",
    )

    def resolve_shp(directory: Path, filename: str, fallbacks: tuple[str, ...]) -> Path:
        candidates = (filename,) + fallbacks
        for name in candidates:
            p = directory / name
            if p.exists():
                return p
        tried = ", ".join(candidates)
        raise FileNotFoundError(f"Under {directory}, none of the following exist: {tried}")

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

    print("Reading shapefiles …")
    id_kw = None if args.city_id_col.strip().lower() == "auto" else args.city_id_col
    g0 = add_unified_city_id(prepare_geoms_wgs84(gpd.read_file(p2000)), id_kw)
    g2 = add_unified_city_id(prepare_geoms_wgs84(gpd.read_file(p2020)), id_kw)
    gs = add_unified_city_id(prepare_geoms_wgs84(gpd.read_file(psub)), id_kw)

    # Shift geometries to 0..360 longitudes to match rasters
    def shift_lon_geom(gdf):
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

    print("Building Core / Expansion / Peri …")
    city_geoms = build_core_expansion_peri(g0, g2, gs, CITY_ID_COL)
    # Attach dissolved 2020 footprint for Expansion intensity denominator
    d2u = dissolve_by_city(g2, CITY_ID_COL)
    d2u = d2u.rename(columns={"geometry": "geometry_2020_full"})
    city_geoms = city_geoms.merge(
        d2u[[CITY_ID_COL, "geometry_2020_full"]], on=CITY_ID_COL, how="left"
    )

    extra_dict: dict[str, xr.DataArray] = extra_arg or {}
    for _name, _da in extra_dict.items():
        if _da.shape != w1.shape:
            raise ValueError(f"extra_fields['{_name}'] shape mismatch vs W1")

    transformer_ag = _transformer_ll_to_cea()
    years_ag = _panel_years_from_w1(w1)

    by_dir = args.out_dir / "by_city"
    by_dir.mkdir(parents=True, exist_ok=True)
    panel_path = args.out_dir / "panel_city_year_region.csv"
    overall_path = args.out_dir / "city_year_overall.csv"
    clim_path = args.out_dir / "city_region_climatology_2000_2021_mean.csv"
    clim_et_path = args.out_dir / "city_region_climatology_ET_2000_2021_mean.csv"
    clim_p_path = args.out_dir / "city_region_climatology_P_2000_2021_mean.csv"
    attrs_path = args.out_dir / "city_attributes.csv"
    attrs_et_path = args.out_dir / "city_attributes_ET.csv"
    attrs_p_path = args.out_dir / "city_attributes_P.csv"
    p_etv_path = args.out_dir / "panel_city_year_region_ET_vegenon.csv"
    p_et_path = args.out_dir / "panel_city_year_region_ET_fullveg.csv"
    p_et_comb_path = args.out_dir / "panel_city_year_region_ET_combined.csv"
    p_p_path = args.out_dir / "panel_city_year_region_P.csv"
    o_etv_path = args.out_dir / "city_year_overall_ET_vegenon.csv"
    o_et_fv_path = args.out_dir / "city_year_overall_ET_fullveg.csv"
    o_et_comb_path = args.out_dir / "city_year_overall_ET_combined.csv"
    o_p_path = args.out_dir / "city_year_overall_P.csv"

    fpanel = foverall = True
    fclim = fclim_et = fclim_p = True
    fattrs = fattrs_et = fattrs_p = True
    fpetv = fp_et = fp_etcomb = fp_ponly = True
    fo_etv = fo_et_fv = fo_etcomb = fo_ponly = True
    panel_columns: list[str] | None = None

    mask_fb: float | None = (
        None
        if args.mask_fallback_threshold < 0
        else float(args.mask_fallback_threshold)
    )
    if mask_fb is not None:
        _fb_desc = "any overlap" if mask_fb <= 0 else str(mask_fb)
        print(
            f"Mask fallback enabled: primary fraction={args.area_threshold}, "
            f"retry with {_fb_desc} when empty"
        )
    else:
        print("Mask fallback disabled (--mask-fallback-threshold -1); expect more NaNs.")

    print(
        "Zonal + city union: per-city computation (master CSV append + by_city/ extras)…"
    )
    for _, geom_row in tqdm(
        city_geoms.iterrows(), total=len(city_geoms), desc="cities"
    ):
        rec_p = records_for_city_region_panel(
            geom_row,
            CITY_ID_COL,
            w1,
            w2,
            v,
            lon_edges,
            lat_edges,
            weight_grid,
            args.area_threshold,
            mask_fb,
            extra_dict,
            transformer_ag,
            years_ag,
        )
        df_p = pd.DataFrame.from_records(rec_p)
        if panel_columns is None:
            panel_columns = df_p.columns.tolist()
            print("  Main panel columns:", ", ".join(map(str, panel_columns)))
        _append_df_csv(panel_path, df_p, first=fpanel)
        fpanel = False
        cid = geom_row[CITY_ID_COL]
        safe_nm = safe_city_filename(cid)
        df_p.to_csv(by_dir / f"{safe_nm}_panel_city_year_region.csv", index=False)

        rec_o = records_for_city_overall_union(
            geom_row,
            CITY_ID_COL,
            w1,
            w2,
            v,
            lon_edges,
            lat_edges,
            weight_grid,
            args.area_threshold,
            mask_fb,
            extra_dict,
            transformer_ag,
            years_ag,
        )
        df_o = pd.DataFrame.from_records(rec_o)
        _append_df_csv(overall_path, df_o, first=foverall)
        foverall = False
        df_o.to_csv(by_dir / f"{safe_nm}_city_year_overall.csv", index=False)

        clim_row = panel_to_region_climatology(df_p, ["W1", "W2", "V"])
        _append_df_csv(clim_path, clim_row, first=fclim)
        fclim = False
        clim_row.to_csv(by_dir / f"{safe_nm}_region_climatology_mean.csv", index=False)

        if "ET" in df_p.columns or "ET_vegenon" in df_p.columns:
            pan_et = df_p.copy()
            et_clim_cols = [c for c in ("ET", "ET_vegenon") if c in pan_et.columns]
            if "ET" in pan_et.columns and "ET_vegenon" in pan_et.columns:
                pan_et["V_ET"] = pan_et["ET"] - pan_et["ET_vegenon"]
                et_clim_cols = ["ET", "ET_vegenon", "V_ET"]
            clim_et_row = panel_to_region_climatology(pan_et, et_clim_cols)
            _append_df_csv(clim_et_path, clim_et_row, first=fclim_et)
            fclim_et = False
            clim_et_row.to_csv(
                by_dir / f"{safe_nm}_region_climatology_ET_mean.csv", index=False
            )

        if "P" in df_p.columns:
            clim_p_row = panel_to_region_climatology(df_p, ["P"])
            _append_df_csv(clim_p_path, clim_p_row, first=fclim_p)
            fclim_p = False
            clim_p_row.to_csv(
                by_dir / f"{safe_nm}_region_climatology_P_mean.csv", index=False
            )

        _base3 = ["city_id", "year", "region"]
        if "ET_vegenon" in df_p.columns:
            _append_df_csv(
                p_etv_path, df_p[_base3 + ["ET_vegenon"]], first=fpetv
            )
            fpetv = False
        if "ET" in df_p.columns:
            _append_df_csv(p_et_path, df_p[_base3 + ["ET"]], first=fp_et)
            fp_et = False
        if {"ET", "ET_vegenon"}.issubset(df_p.columns):
            pet = df_p[_base3 + ["ET", "ET_vegenon"]].copy()
            pet["V_ET"] = pet["ET"] - pet["ET_vegenon"]
            _append_df_csv(p_et_comb_path, pet, first=fp_etcomb)
            fp_etcomb = False
        if "P" in df_p.columns:
            _append_df_csv(p_p_path, df_p[_base3 + ["P"]], first=fp_ponly)
            fp_ponly = False

        _by = ["city_id", "year"]
        if "ET_vegenon" in df_o.columns:
            _append_df_csv(
                o_etv_path, df_o[_by + ["ET_vegenon"]], first=fo_etv
            )
            fo_etv = False
        if "ET" in df_o.columns:
            _append_df_csv(o_et_fv_path, df_o[_by + ["ET"]], first=fo_et_fv)
            fo_et_fv = False
        if {"ET", "ET_vegenon"}.issubset(df_o.columns):
            oet = df_o[_by + ["ET", "ET_vegenon"]].copy()
            oet["V_ET"] = oet["ET"] - oet["ET_vegenon"]
            _append_df_csv(o_et_comb_path, oet, first=fo_etcomb)
            fo_etcomb = False
        if "P" in df_o.columns:
            _append_df_csv(o_p_path, df_o[_by + ["P"]], first=fo_ponly)
            fo_ponly = False

        attr_row = compute_city_attributes_one(
            geom_row,
            df_o,
            CITY_ID_COL,
            args.year_early_end,
            args.year_late_start,
            transformer_ag,
        )
        _append_df_csv(attrs_path, pd.DataFrame([attr_row]), first=fattrs)
        fattrs = False
        pd.DataFrame([attr_row]).to_csv(
            by_dir / f"{safe_nm}_city_attributes.csv", index=False
        )

        attrs_et_part = compute_city_attributes_et(
            df_o,
            CITY_ID_COL,
            year_early_end=args.year_early_end,
            year_late_start=args.year_late_start,
        )
        if attrs_et_part is not None and len(attrs_et_part):
            if "Expansion" in attr_row:
                attrs_et_part = attrs_et_part.copy()
                attrs_et_part["Expansion"] = attr_row["Expansion"]
            _append_df_csv(attrs_et_path, attrs_et_part, first=fattrs_et)
            fattrs_et = False
            attrs_et_part.to_csv(
                by_dir / f"{safe_nm}_city_attributes_ET.csv", index=False
            )

        attrs_p_part = compute_city_attributes_p(
            df_o,
            CITY_ID_COL,
            year_early_end=args.year_early_end,
            year_late_start=args.year_late_start,
        )
        if attrs_p_part is not None and len(attrs_p_part):
            if "Expansion" in attr_row:
                attrs_p_part = attrs_p_part.copy()
                attrs_p_part["Expansion"] = attr_row["Expansion"]
            _append_df_csv(attrs_p_path, attrs_p_part, first=fattrs_p)
            fattrs_p = False
            attrs_p_part.to_csv(
                by_dir / f"{safe_nm}_city_attributes_P.csv", index=False
            )

    print("Written (append-by-city master):", panel_path)
    print("Written (append-by-city master):", overall_path)
    print("Written (append-by-city master):", clim_path)
    if not fclim_et:
        print("Written (append-by-city master):", clim_et_path)
    if not fclim_p:
        print("Written (append-by-city master):", clim_p_path)
    print("Written (append-by-city master):", attrs_path)
    if not fattrs_et:
        print("Written (append-by-city master):", attrs_et_path)
    if not fattrs_p:
        print("Written (append-by-city master):", attrs_p_path)
    for label, flag, pth in (
        ("ET_vegenon zonal", fpetv, p_etv_path),
        ("ET zonal", fp_et, p_et_path),
        ("ET combined zonal", fp_etcomb, p_et_comb_path),
        ("P zonal", fp_ponly, p_p_path),
        ("ET_vegenon union", fo_etv, o_etv_path),
        ("ET union", fo_et_fv, o_et_fv_path),
        ("ET combined union", fo_etcomb, o_et_comb_path),
        ("P union", fo_ponly, o_p_path),
    ):
        if not flag:
            print(f"Written (append-by-city master) [{label}]:", pth)
    print("Per-city CSV directory:", by_dir)

    meta_path = args.out_dir / "RUN_METADATA.txt"
    meta_path.write_text(
        "V = W1 - W2 contrasts vegetation scenarios on ET/P — not a causal attribution.\n"
        "Urban structure encoded via core/expansion/peri and Expansion metrics.\n"
        "Annual grids ET / ET_vegenon / P join the panel_city_year_region table when available "
        "(plus ancillary panel_city_year_region_ET*_ and city_year_overall_* CSVs).\n"
        "Writer pattern: appended master CSVs plus per-city files under by_city/.\n"
        f"W1: {args.w1}\nW2: {args.w2}\n"
        f"ET: {args.et_nc}\nET_vegenon: {args.et_vegenon_nc}\nP: {args.precip_nc}\n"
        f"interp_res_deg(°): {args.interp_res_deg if args.interp_res_deg is not None else 'none'}\n"
        f"Primary overlap fraction threshold: {args.area_threshold}\n"
        f"Fallback fraction threshold: {args.mask_fallback_threshold} (-1 disables)\n"
        f"Weights: cos(lat) proportional to grid-cell area scaling\n",
        encoding="utf-8",
    )
    print("Done.")


if __name__ == "__main__":
    main()
